import argparse

def _get_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path for input question dataset",
    )
    parser.add_argument(
        "--scaling_strategy",
        type=str,
    )
    parser.add_argument(
        "--question_type",
        type=str,
        choices=[
            'None',  # All types
            # Original SAT dataset types
            'action_conseq', 'ego_movement', 'goal_aim', 'obj_movement', 'perspective', 'action_sequence', 'action_consequence',
            # MMSI-Bench question types
            'motion_(cam.)', 'motion_(obj.)',
            'positional_relationship_(cam.-obj.)', 'positional_relationship_(reg.-reg.)', 'positional_relationship_(obj.-obj.)',
            'positional_relationship_(cam.-cam.)', 'positional_relationship_(cam.-reg.)', 'positional_relationship_(obj.-reg.)',
            'action_consequence', 'goal_aim',
            'attribute_(appr.)', 'attribute_(meas.)', 'msr'
        ],
        # default='None',
        required=True,
        help= "None represents all types. Includes both SAT dataset and MMSI-Bench question types.",
        )
    parser.add_argument(
        "--sampling_interval_angle",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--sampling_interval_meter",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--fixed_rotation_magnitudes",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--fixed_forward_magnitudes",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--max_steps_per_question",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_tries_gpt",
        type=int,
        default=5,
        required=True,
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=500,
        required=True,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"], 
        default="val",
        required=True,
    )

    parser.add_argument(
        "--max_turn_angle",
        type=float,
        default=60.0
    )

    parser.add_argument(
        "--max_forward_distance",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--num_top_candidates",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max_inference_batch_size",
        type=int,
        default=3
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=2
    )

    parser.add_argument(
        "--helpful_score_threshold",
        type=int,
        default=8
    )

    parser.add_argument(
        "--exploration_score_threshold",
        type=int,
        default=8
    )

    parser.add_argument(
        "--vlm_model_name",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4.1", "o4-mini", "o1", "OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B", "OpenGVLab/InternVL3_5-8B", "OpenGVLab/InternVL3_5-14B", 'Qwen/Qwen3-VL-8B', 'Qwen/Qwen3-VL-14B'],
    )

    parser.add_argument(
        "--vlm_qa_model_name",
        type=str,
        default=None,
        choices=[None, "None", "gpt-4o", "gpt-4.1", "o4-mini", "o1", "OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-14B", "OpenGVLab/InternVL3_5-8B", "OpenGVLab/InternVL3_5-14B", 'Qwen/Qwen3-VL-8B', 'Qwen/Qwen3-VL-14B'],
    )

    parser.add_argument(
        "--num_question_chunks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--question_chunk_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        '--camera_mixed',
        action='store_true',
        default=False,
    )


def _get_svc_args(parser):
    parser.add_argument(
        "--task",
        type=str,
        default="img2trajvid_s-prob"
        )
        
    parser.add_argument(
        "--replace_or_include_input",
        type=bool,
        default=True
        )
        
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.0
        )
        
    parser.add_argument(
        "--guider",
        type=int,
        default=1
        )
        
    parser.add_argument(
        "--L_short",
        type=int,
        default=576
        )
        
    parser.add_argument(
        "--num_targets",
        type=int,
        default=8
        )
        
    parser.add_argument(
        "--use_traj_prior",
        type=bool,
        default=True
        )
        
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        default="interp"
        )

def get_svc_args():
    
    parser = argparse.ArgumentParser(description="Simple example of Pipeline args.")
    _get_svc_args(parser)
    _get_pipeline_args(parser)
    
    # Add verifier-specific arguments (optional, for verifier pipeline)
    parser.add_argument('--enable_verifier', action='store_true', default=False,
                        help='Enable VLM verifier (default: False)')
    parser.add_argument('--verification_threshold', type=float, default=0.7,
                        help='Verification threshold for claim acceptance rate (default: 0.7)')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline mode without verifier')
    parser.add_argument('--no_answer_choices', action='store_true', default=False,
                        help='Remove answer choices from helpfulness and exploration prompts (default: False)')
    parser.add_argument('--use_top_k_filtering', action='store_true', default=False,
                        help='Use top-k filtering instead of threshold-based filtering for helpfulness and exploration beams (default: False)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of top items to select when using top-k filtering (default: 2)')
    parser.add_argument('--use_validated_claims_in_qa', action='store_true', default=False,
                        help='Include validated claims in the final QA prompt for better reasoning (default: False)')
    parser.add_argument('--voting-inference', action='store_true', default=False,
                        help='Use voting mechanism with multiple independent VQA queries instead of single multi-image query (default: False)')
    parser.add_argument('--compute_uncertainty', action='store_true', default=False,
                        help='Compute uncertainty metrics using log probabilities (default: False)')

    return parser.parse_args()