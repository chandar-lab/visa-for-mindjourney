# A VLM wrapper for adapting both close-source (i.e. API) and open-source (i.e. local) models.
from math import log
import torch
from utils.api import ChatAPI, AzureConfig
from utils.InternVL3 import *
from utils.prompt_formatting import *
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

class VLMWrapper:
    def __init__(self, model_name, qa_model_name=None):
        if model_name in ['gpt-4o', 'gpt-4.1', 'o4-mini', 'o1']:
            api_info = {
                "gpt-4o": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.01,
                },
                "gpt-4.1": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.005,
                },
                "o4-mini": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.000375,
                },
                "o1": {
                    "api_version": "2024-12-01-preview",
                    "api_price": 0.005,
                }
            }
            # Assume we use the Azure OpenAI API
            config = AzureConfig(model_name, api_info[model_name]["api_version"], api_info[model_name]["api_price"]) # already set greedy decoding in the config
            self.model = ChatAPI(config)
            self.qa_model = None
            if qa_model_name not in (None, "None") and qa_model_name != model_name:
                qa_config = AzureConfig(qa_model_name, api_info[qa_model_name]["api_version"], api_info[qa_model_name]["api_price"])
                self.qa_model = ChatAPI(qa_config)
            self.prompt_style = 'gpt'
        elif model_name in ['OpenGVLab/InternVL3-8B', 'OpenGVLab/InternVL3-14B', 'OpenGVLab/InternVL3_5-8B', 'OpenGVLab/InternVL3_5-14B']:
            assert qa_model_name in (None, "None") or qa_model_name == model_name, "Separate Score/QA model is not supported for InternVL3."
            # device_map = split_model(model_name)
            device_map = "cuda:1"
            device_map = split_model(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map).eval()
            # self.model = AutoModel.from_pretrained(
            #     model_name,
            #     torch_dtype=torch.bfloat16,
            #     load_in_8bit=False,
            #     low_cpu_mem_usage=True,
            #     use_flash_attn=True,
            #     trust_remote_code=True,
            #     device_map=device_map).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            self.generation_config = dict(max_new_tokens=1024, do_sample=False) # greedy decoding
            self.prompt_style = 'internvl3'
        elif model_name in ['Qwen/Qwen3-VL-8B', 'Qwen/Qwen3-VL-14B']:
            assert qa_model_name in (None, "None") or qa_model_name == model_name, "Separate Score/QA model is not supported for Qwen3-VL."

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.prompt_style = 'qwen3-vl'
            self.qa_model = None  # Qwen3-VL doesn't support separate QA models
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        self.curr_prompt = None
    
    def format_prompt(self, prompt_type, question, answer_choices, images, action_consequences=None, sys_prompt=None, **kwargs):
        if prompt_type == 'bounding_box':
            return format_spatial_vqa_prompt_bbox(question=question, answer_choices=answer_choices, images=images)
        if prompt_type == 'answer_baseline':
            return format_spatial_vqa_prompt_answer_baseline(question=question, answer_choices=answer_choices, images=images)
        elif prompt_type == 'answer_scaling':
            return format_spatial_vqa_prompt_answer_scaling(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences, action_claims=kwargs.get('action_claims', None))
        elif prompt_type == "prompt_scores":
            return format_spatial_vqa_prompt_scores(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences, sys_prompt=sys_prompt)
        elif prompt_type == "answer_baseline_fill":
            return format_spatial_vqa_prompt_answer_baseline_fill_in_blank(question=question, answer_choices=answer_choices, images=images)
        elif prompt_type == "answer_scaling_fill":
            return format_spatial_vqa_prompt_answer_scaling_fill_in_blank(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences)
        elif prompt_type == "prompt_scores_fill":
            return format_spatial_vqa_prompt_scores_fill_in_blank(question=question, answer_choices=answer_choices, images=images, action_consequences=action_consequences, sys_prompt=sys_prompt)
        elif prompt_type == "answer_baseline_open_ended":
            return format_spatial_vqa_prompt_answer_baseline_open_ended(question=question, images=images)
        elif prompt_type == "answer_scaling_open_ended":
            return format_spatial_vqa_prompt_answer_scaling_open_ended(question=question, images=images, action_consequences=action_consequences, action_claims=kwargs.get('action_claims', None))
        elif prompt_type == "claim_generation":
            return format_verification_prompt_claim_generation(
                action_description=kwargs.get('action_description', ''),
                frames=images,
                frame_range=kwargs.get('frame_range', ''),
                question=question
            )
        elif prompt_type == "claim_verification":
            return format_verification_prompt_claim_verification(
                claim=kwargs.get('claim', {}),
                frames=images
            )
        else:
            raise ValueError(f"Prompt type {prompt_type} is not supported.")
    
    def run_prompt(self, prompt_type, system_prompt, content, return_log_probs=False, **kwargs):
        self.curr_prompt = {"system": system_prompt, "content": content}
        if self.prompt_style == 'gpt':
            content = format_gpt_content(content)
            if prompt_type[:7] == 'answer_' and self.qa_model is not None:
                response = self.qa_model.get_system_response_with_content(system_prompt, content)
            else:
                response = self.model.get_system_response_with_content(system_prompt, content)
            return response
        elif self.prompt_style == 'internvl3':
            # Implement log-likelihood scoring for answer choices
            # This provides true probability estimates for each answer choice
            # by computing log-likelihood scores using the model's forward pass
            answer_choices = kwargs.get('answer_choices', [])
            if not return_log_probs:
                # Fallback to regular response if no answer choices provided
                content = format_internvl3_content(content, "cuda:1")
                question = content['question']
                pixel_values = content['pixel_values']
                num_patches_list = content['num_patches_list']
                response, history = self.model.chat(self.tokenizer, pixel_values, system_prompt+'\n'+question, self.generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
                return response
            else:
                # Get log-likelihood scores and entropies for each answer choice
                results = self._get_answer_log_likelihoods_internvl3(
                    system_prompt, content.copy(), answer_choices, teacher_forcing=False
                )
                answer_log_likelihoods = results['log_likelihoods']
                token_level_entropies = results['entropies']
                
                # Get regular response for the text
                content = format_internvl3_content(content, "cuda:1")
                question = content['question']
                pixel_values = content['pixel_values']
                num_patches_list = content['num_patches_list']                

                # Calculate Uncertainty across the different answer choices
                # convert the log-likelihoods of all answer choices to probabilities, then calculate entropy over this distribution
                # Measures how evenly distributed the model's confidence is across the different answer options
                """
                Imagine we have 3 answer choices with these log-likelihoods:
                    Choice A: -1.0 (highest confidence)
                    Choice B: -2.0
                    Choice C: -3.0 (lowest confidence)
                Token-level entropy would measure: "How uncertain is the model when generating each token of Choice A?"
                Answer-level entropy would measure: "How uncertain is the model about which answer choice is correct?" (In this case, it would be relatively low because Choice A is clearly preferred)
                
                Token-level entropy: Helps identify if the model struggles with specific parts of an answer
                Answer-level entropy: Helps identify if the model is confident about which answer is correct overall
                """
                log_likelihoods = list(answer_log_likelihoods.values())
                answer_level_entropy = None
                if log_likelihoods and all(ll != float('-inf') for ll in log_likelihoods):
                    max_log_likelihood = max(log_likelihoods)
                    normalized_log_likelihoods = [ll - max_log_likelihood for ll in log_likelihoods]
                    probs = F.softmax(torch.tensor(normalized_log_likelihoods), dim=0)
                    answer_level_entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                return {'log_probs': answer_log_likelihoods, 'token_entropy': token_level_entropies, 'answer_entropy': answer_level_entropy}
        elif self.prompt_style == 'qwen3-vl':
            
            messages = format_qwen3vl_content(content, "cuda:1")
            
            # Add system prompt to the first message
            if messages and len(messages) > 0:
                if "content" in messages[0] and isinstance(messages[0]["content"], list):
                    # Add system prompt as text at the beginning
                    messages[0]["content"].insert(0, {"type": "text", "text": system_prompt + "\n\n"})
                else:
                    # If content is not a list, wrap it
                    messages[0]["content"] = [
                        {"type": "text", "text": system_prompt + "\n\n"},
                        {"type": "text", "text": messages[0]["content"]}
                    ]
            else:
                # If no messages, create one with system prompt
                messages = [{"role": "user", "content": [{"type": "text", "text": system_prompt}]}]
            
            # Process vision information
            images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16)
            
            # Prepare inputs
            inputs = self.processor(
                text=messages, 
                images=images, 
                videos=videos, 
                do_resize=False, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False,  # greedy decoding
                temperature=None
            )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
        else:
            raise ValueError(f"Prompt style {self.prompt_style} is not supported.")
    
    def _get_answer_log_likelihoods_internvl3(self, system_prompt, content, answer_choices, teacher_forcing=True):
        """Calculate log-likelihood and entropy for each answer choice using InternVL-3 model."""
        
        # Format content
        formatted_content = format_internvl3_content(content, "cuda:1")
        question = formatted_content['question']
        pixel_values = formatted_content['pixel_values']
        num_patches_list = formatted_content['num_patches_list']
        total_patches = sum(num_patches_list) if num_patches_list else 0

        answer_log_likelihoods = {}
        answer_entropies = {}
        
        for choice in answer_choices:
            try:
                # Create base prompt
                base_prompt = system_prompt + '\n' + question + "\nAnswer:"
                base_inputs = self.tokenizer(base_prompt, return_tensors="pt", padding=True)
                base_input_ids = base_inputs["input_ids"].to(self.model.device)
                
                # Tokenize the choice
                choice_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                if not choice_tokens:
                    answer_log_likelihoods[choice] = float('-inf')
                    answer_entropies[choice] = float('inf')
                    continue
                
                # Create full sequence: base_prompt + choice
                full_input_ids = torch.cat([
                    base_input_ids, 
                    torch.tensor([choice_tokens]).to(self.model.device)
                ], dim=1)
                
                # Create image_flags tensor (this was missing!)
                # image_flags indicates which tokens correspond to images
                image_flags = torch.ones(total_patches, 1, dtype=torch.long).to(self.model.device)
                
                # Use the model's forward method with proper parameters
                with torch.no_grad():
                    try:
                        outputs = self.model.forward(
                            pixel_values=pixel_values,
                            input_ids=full_input_ids,
                            image_flags=image_flags,
                            return_dict=True
                        )
                        
                        print("model.forward() called successfully")
                        
                        # Extract logits
                        logits = outputs.logits
                        
                        # Calculate log-likelihood and entropy for answer tokens
                        answer_start_idx = base_input_ids.shape[1]
                        answer_logits = logits[0, answer_start_idx:answer_start_idx + len(choice_tokens), :]
                        
                        # Calculate log probabilities
                        log_probs = F.log_softmax(answer_logits, dim=-1)
                        probs = F.softmax(answer_logits, dim=-1)
                        
                        # Sum log-likelihoods and entropies
                        total_log_likelihood = 0.0
                        total_entropy = 0.0
                        
                        for i, token_id in enumerate(choice_tokens):
                            if i < log_probs.shape[0]:
                                total_log_likelihood += log_probs[i, token_id].item()
                                token_entropy = -torch.sum(probs[i, :] * torch.log(probs[i, :] + 1e-10))
                                total_entropy += token_entropy.item()
                        
                        answer_log_likelihoods[choice] = total_log_likelihood
                        answer_entropies[choice] = total_entropy
                        
                    except Exception as forward_error:
                        print(f"Forward method failed for choice '{choice}': {forward_error}")
                        
                        
            except Exception as e:
                print(f"Warning: Error calculating log-likelihood and entropy for choice '{choice}': {e}")
                answer_log_likelihoods[choice] = float('-inf')
                answer_entropies[choice] = float('inf')
        
        return {
            'log_likelihoods': answer_log_likelihoods,
            'entropies': answer_entropies
        }