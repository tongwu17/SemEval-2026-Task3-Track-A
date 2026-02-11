import argparse
from os.path import abspath, dirname, exists, join
import time
from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content
from tqdm import tqdm

from utils_fewshot import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from utils_data import flat_to_task_a_format_iter, iter_jsonl, iter_src_filepaths, load_samples_flat, write_jsonl


def predict_va_with_bulkchain(samples, model_name, provider_path, max_retries=3, batch_size=10, sleep_time=None, **kwargs):
    print(f"[INFO] Using model: {model_name} from `{provider_path}`")
    print(f"[INFO] Total samples to predict: {len(samples)}")

    content_it = iter_content(
        # 1. Your schema.              
        schema=[
            {
                "prompt": FEW_SHOT_EXAMPLES + "\nNow predict: Text:\n{text} Aspect: {aspect}\n\nOutput ONLY the scores in format valence#arousal:", 
                "out": "va"
            },
        ],
        # 2. Your third-party model implementation.
        llm=dynamic_init(class_filepath=provider_path)(
            model_name=model_name,
            assistant=SYSTEM_PROMPT,
            **kwargs),
        # 3. Customize your inference and result providing modes: 
        infer_mode="batch_async", 
        batch_size=batch_size,
        # 4. Your iterator of dictionaries
        input_dicts_it=tqdm(samples, desc=f"Predicting `{model_name}`"),
        attempts=max_retries,
    )

    for batch in content_it:
        assert(isinstance(batch, list))
        for record in batch:
            yield record

        if sleep_time is not None:
            time.sleep(sleep_time)


if __name__ == "__main__":

    # We can register other dataset templates here if needed. 
    # Each template based on the following variables:
    # 1. language 
    # 2. domain
    cur_dir = dirname(abspath(__file__))
    datasets = {
        "eval_20": join(cur_dir, "../task-dataset-split/{lang}/{lang}_{domain}_train_alltasks_20_without_va.jsonl")
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', type=str, default=None, help='Model name')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--provider_path', type=str, default="replicate_104.py", help='Provider path')
    parser.add_argument('--api_token', type=str, default=None, help='API key')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--sleep_time', type=int, default=None, help='Sleep time')
    parser.add_argument('--dataset_name', type=str, default="eval_20", help='Dataset name', choices=list(datasets.keys()))
    parser.add_argument('--langs', nargs="*", type=str, default=["eng"], help='Language')
    parser.add_argument('--domains', nargs="*", type=str, default=["restaurant", "laptop", "finance", "hotel"], help='Domain')
    args = parser.parse_args()

    template = datasets[args.dataset_name]

    args_dict = dict(vars(args))
    del args_dict['dataset_name']
    del args_dict['langs']
    del args_dict['domains']

    for lang, domain, src_path in iter_src_filepaths(langs=args.langs, domains=args.domains, template=template):

        dataset_tag = '20' if args.dataset_name == 'eval_20' else args.dataset_name
        result_template = f"pred_{lang}_{domain}_{dataset_tag}_{args.model_name.split('/')[-1]}.jsonl"
        output_path = f"{args.dataset_name}/flat/{result_template}"

        if not exists(output_path):
            results_it = predict_va_with_bulkchain(samples=load_samples_flat(src_path), **args_dict)
            write_jsonl(results_it=results_it, output_path=output_path)
        else:
            print(f"[INFO] Output file already exists: {output_path}. Skipping...")
            pass

        # Convert to evaluation submission
        write_jsonl(
            results_it=flat_to_task_a_format_iter(iter_jsonl(output_path)), 
            output_path=f"{args.dataset_name}/{result_template}"
        )