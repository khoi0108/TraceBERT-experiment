import logging
import multiprocessing
import os
import sys

sys.path.append("..")
sys.path.append("../..")

from code_search.single.single_train import train_single_iteration
from code_search.twin.twin_train import get_train_args, init_train_env, train
from data_process import DataProcess
from common.data_structures import Examples
from common.models import TBertT, TBertI, TBertI2

logger = logging.getLogger(__name__)


def read_OSS_examples(data_dir):
    commit_file = os.path.join(data_dir, "commit_file")
    issue_file = os.path.join(data_dir, "issue_file")
    link_file = os.path.join(data_dir, "link_file")
    examples = []
    data_processor = DataProcess(False) 
    issues = data_processor.read_OSS_artifacts(issue_file, "issue", clean=True)
    commits = data_processor.read_OSS_artifacts(commit_file, "commit", clean=True)
    links = data_processor.read_OSS_artifacts(link_file, "link", clean=True)
    
    if not (isinstance(issues, dict) and isinstance(commits, dict)):
        return examples
    
    if not isinstance(links, list):
        return examples

    for lk in links:
        iss = issues[lk[0]]
        cm = commits[lk[1]]
        # join the tokenized content
        iss_text = f"{iss['issue_desc']} {iss['issue_comments']}"
        cm_text = f"{cm['summary']} [A] {cm['diff_added']} [/A] [D] {cm['diff_removed']} [/D]"
        example = {
            "NL": iss_text,
            "PL": cm_text
        }
        examples.append(example)
    return examples


def load_examples(data_dir, model, num_limit):
    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    logger.info("Creating examples from dataset file at {}".format(data_dir))
    raw_examples = read_OSS_examples(data_dir)
    if num_limit:
        raw_examples = raw_examples[:num_limit]
    examples = Examples(raw_examples)
    if isinstance(model, TBertT) or isinstance(model, TBertI2) or isinstance(model, TBertI):
        examples.update_features(model, multiprocessing.cpu_count())
    return examples


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type='single')
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    train_examples = load_examples(train_dir, model=model, num_limit=args.train_num)
    valid_examples = load_examples(valid_dir, model=model, num_limit=args.valid_num)
    if not train_examples or not valid_examples:
        logger.info("Example loading error.")    
        return
    train(args, train_examples, valid_examples, model, train_single_iteration)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
