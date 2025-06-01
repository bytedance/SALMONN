import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool



PROMPT_TEMPLATE = """
Given the following inputs:

1. **Ground Truth Video Caption**: {caption}
2. **Question Related to the Caption**: {question}
3. **Ground Truth Answer**: {answer}
4. **Model Predicted Answer**: {prediction}

Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the video caption and the question. Consider the following criteria for evaluation:

- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the video caption?
- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?
- **Clarity**: Assess the clarity of the predicted answer. Look for issues such as repetition, unclear descriptions, or any grammatical errors that could hinder understanding.
- **Completeness**: Determine if the predicted answer fully covers the scope of the ground truth answer. Does it leave out critical information or does it include all necessary details?

**Output Format**:
Please generate the response in the form of a string that represents a Python dictionary, with keys Score and Explanation. The Score should be an integer or a float from 1 to 5, and Explanation should provide a brief judgement of the prediction.

For example, your response should look like this: "'{'Score': 4.8, 'Explanation': 'The prediction accurately reflects the ground truth with minor discrepancies.'}".
"""

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    parser.add_argument("--api_key", required=True, type=str, help="OpenAI API key")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI API base")
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--gpt_version", default="0613", type=str, help="GPT version")

    args = parser.parse_args()
    return args

def shrink_string_correctly(text):
    # Split the text into sentences for better analysis
    parts = text.split(" ")
    output = []

    for part in parts:
        # Check if the sentence (or part) has been seen before
        if part not in output:
            output.append(part)
        elif part == output[-1]:
            # Stop adding once a duplicate is found since the example suggests stopping at the first repeat occurrence
            break

    # Reconstruct the string, taking into account the removal of duplicates
    return " ".join(output)


def annotate(prediction_set, caption_files, output_dir, gpt_version):
    """
    Evaluates question and answer pairs using GPT-3 and
    returns a score for temporal understanding.
    """
    # import pdb; pdb.set_trace()
    for file in caption_files:
        key = file[:-5]  # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set["question"]
        answer = qa_set["answer"]
        pred = qa_set["pred"]
        caption = qa_set["caption"]

        # pred = shrink_string_correctly(pred)

            
        try:
            print(key, "query")
            if pred == "" or len(pred) < 2:
                result_qa_pair = [{"score": 0}, qa_set]
                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f, indent=4)
                continue
            # Compute the temporal understanding score
            # import pdb; pdb.set_trace()
            completion = openai.ChatCompletion.create(
                model=f"gpt-3.5-turbo-{gpt_version}",
                messages=[
                    {
                        "role": "user",
                        "content": "Given the following inputs:\n"
                        f"1. **Ground Truth Video Caption**: {caption}\n"
                        f"2. **Question Related to the Caption**: {question}\n"
                        f"3. **Ground Truth Answer**: {answer}\n"
                        f"4. **Model Predicted Answer**: {pred}\n\n"
                        "Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the video caption and the question. Consider the following criteria for evaluation:\n\n"
                        "- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the video caption?\n"
                        "- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?\n"
                        "- **Clarity**: Assess the clarity of the predicted answer. Look for issues such as repetition, unclear descriptions, or any grammatical errors that could hinder understanding.\n"
                        "- **Completeness**: Determine if the predicted answer fully covers the scope of the ground truth answer. Does it leave out critical information or does it include all necessary details?\n\n"
                        "**Output Format**:\n"
                        "Please generate the response in the form of a string that represents a Python dictionary, with keys Score and Explanation. The Score should be an integer or a float from 1 to 5, and Explanation should provide a brief judgement of the prediction.\n\n"
                        "For example, your response should look like this: {'score': 4.8, 'explanation': 'The prediction accurately reflects the ground truth with minor discrepancies.'}\n",
                    },
                ],
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            print(key, "done")
            # import pdb; pdb.set_trace()

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key.split('/')[-1]}.json", "w") as f:
                json.dump(result_qa_pair, f, indent=4)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")



def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # import pdb; pdb.set_trace()
    if args.num_chunks > 1:
        pred_contents = []
        for _idx in range(args.num_chunks):
            if args.output_name != "":
                file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}_{args.output_name}.json")
            else:
                file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}.jsonl")   
            try:
                for line in open(file):
                    pred_contents += [json.loads(line)]
            except:
                # import pdb; pdb.set_trace()
                continue
            

    else:
        pred_contents = [json.loads(line) for line in open(args.pred_path)]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        # import pdb;pdb.set_trace()
        if "modal_path" in sample:
            sample["video_name"] = sample["modal_path"]
            sample["question"] = sample["query"]
            sample["pred"] = sample["model_prediction"]["message"]
        video_id = sample["video_name"].split("/")[-1] 

        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample["video_name"] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x["video_name"] for x in new_pred_contents]
    # import pdb; pdb.set_trace()
    caption_files = [f"{id.split('/')[-1]}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    # import pdb; pdb.set_trace()
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample["video_name"]
        question = sample["question"]
        answer = sample["answer"]
        pred = sample["pred"]
        caption = sample["caption"] if "caption" in sample else ""
        qa_set = {"question": question, "answer": answer, "pred": pred, "caption": caption}
        prediction_set[id] = qa_set

    # Set the OpenAI API key.
    openai.api_key = args.api_key  # Your API key here
    if args.api_base:
        openai.api_base = args.api_base  # Your API base here
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            # import pdb; pdb.set_trace()
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")
            # import pdb; pdb.set_trace()

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i : i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args.gpt_version) for part in all_parts]
            print("Generate", len(all_parts), "subprocess.")

            # import pdb; pdb.set_trace()
            # Use a pool of workers to process the files in parallel.
            # with Pool() as pool:
            #     pool.starmap(annotate, task_args)
            # import pdb; pdb.set_trace()
            annotate(task_args[0][0], task_args[0][1], task_args[0][2], task_args[0][3])

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                if "gpt4v" in args.pred_path:
                    if "sorry" in content[1]["pred"].lower():
                        continue
                combined_contents[file_name[:-5]] = content


    # Calculate average score
    score_sum = 0
    count = 0
    acc = 0
    # import pdb;pdb.set_trae
    for key, result in combined_contents.items():
        count += 1
        try:
            # key = result[0].keys()[0]
            # import pdb; pdb.set_trace()
            for _ in result[0].keys():
                score_match = result[0][_]
                score = int(score_match)
                score_sum += score
                if score >=3.0:
                    acc += 1
                break
        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            import pdb; pdb.set_trace()
    average_score = score_sum / count
    acc = acc / count *100
    acc = f"{acc:.2f}/{average_score:.2f}"
    combined_contents["average_score"] = average_score
    combined_contents["result"] = acc
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file, indent=4)
    print("Average score:", average_score)
    print("Accuracy:", acc)
    print("Valid question nunber:", count)


if __name__ == "__main__":
    main()
