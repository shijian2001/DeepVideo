import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_answer(solution):
    """Extract answer from <answer></answer> tags"""
    match = re.search(r'<answer>(.*?)</answer>', solution)
    if match:
        return match.group(1).strip()
    return solution


def build_system_message(example):
    """Build system content for video processing"""
    if example['data_type'] == 'video':
        return (
            'You are a helpful assistant.\n\n# Tools\n'
            'You may call one or more functions to assist with the user query.\n'
            'You are provided with function signatures within <tools></tools> XML tags:\n'
            '<tools>\n'
            '{"type":"function","function":{"name":"video_zoom_in_tool","description":"Zoom in on a specific segment of a video by trimming it based on a start and end time.","parameters":{"type":"object","properties":{"start_time":{"type": int, "description":"Start time of the video segment in seconds."},"end_time":{"type":"int","description":"End time of the video segment in seconds."}},"required":["start_time","end_time"]}}}\n'
            '</tools>\n\n'
            '# How to call a tool\n'
            'Return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>\n\n'
            '**Example**:\n'
            '<tool_call>\n'
            '{"name": "video_zoom_in_tool", "arguments": {"start_time": 5.0, "end_time": 12.5}}\n'
            '</tool_call>'
        )
    elif example['data_type'] == 'image':
        return "You are a helpful assistant"


def build_usr_message(example):
    """Build question content with proper formatting"""
    import copy

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "\nOptions:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']

    raw_question = copy.deepcopy(question)

    # Add media tag at the beginning
    if example['data_type'] == 'video':
        question = "<video>\n" + question
    elif example['data_type'] == 'image':
        question = "<image>\n" + question

    content = QUESTION_TEMPLATE.format(Question=question)

    if example['data_type'] == 'video':
        content += "Think first, call **video_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>."
    elif example['data_type'] == 'image':
        content += "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."

    content += TYPE_TEMPLATE.get(example['problem_type'], "")

    return content, raw_question


def process_fn(example, idx):
    """Process function to transform data format"""
    data_type = example["data_type"]
    problem_type = example["problem_type"]
    problem_id = example["problem_id"]
    path = example["path"]
    solution = example["solution"]

    system_message = build_system_message(example)
    usr_message, raw_question = build_usr_message(example)
    answer = extract_answer(solution)

    data = {
        "data_source": "video-r1",
        "prompt": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": usr_message
            }
        ],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "question_type": problem_type,
            "data_type": data_type,
            "index": problem_id,
            "answer": answer,
            "question": raw_question
        },
    }

    if data_type == "video":
        data["env_name"] = "video_toolbox"
        data["videos"] = [{"video": path}]
        data["images"] = None
    elif data_type == "image":
        data["env_name"] = ""
        data["images"] = [{"path": path}]
        data["videos"] = None
    else:
        raise ValueError("Invalid data type")

    # Add split if available
    if "split" in example:
        data["extra_info"]["split"] = example["split"]

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input JSON file")
    parser.add_argument("--local_dir", default="~/data/processed_data")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--output_name", default="processed_data", help="Output file name prefix")

    args = parser.parse_args()

    # Load dataset from JSON file
    dataset = datasets.load_from_disk(args.input_file) if os.path.isdir(
        args.input_file) else datasets.Dataset.from_json(args.input_file)

    # Process dataset
    processed_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names  # Remove original columns
    )

    # Prepare output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save processed dataset
    output_file = os.path.join(local_dir, f"{args.output_name}.parquet")
    processed_dataset.to_parquet(output_file)

    print(f"Processed {len(processed_dataset)} examples")
    print(f"Saved to: {output_file}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

    # Print sample for verification
    print("\nSample processed data:")
    print(processed_dataset[0])
    print(processed_dataset[10])
