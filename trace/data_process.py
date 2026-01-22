# type: ignore
import pandas as pd
from tqdm import tqdm
import logging
import configparser
import re
import sys
import numpy as np
import random
import argparse
import os
import json
import csv
import emoji

from git_repo_collector import GitRepoCollector, Commits, Issues, Links
import nltk

nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class DataProcess:

    def __init__(self, training) -> None:
        self.training = training

    def __save_artifacts(self, art_list, output_file):
        df = pd.DataFrame(art_list)
        df.to_csv(output_file, index=True)

    def read_OSS_artifacts(self, file_path, type, artifact = None, clean=False):        
        df = pd.read_csv(file_path, keep_default_na=False)
        if type == "commit":
            artifact = Commits()
            for index, row in df.iterrows():
                artifact.add_commit(commit_id=row["commit_id"], summary=row["summary"], diff_added=row["diff_added"], 
                                    diff_removed=row["diff_removed"], files=row["files"], commit_time=row["commit_time"])
            commit_list = artifact.get_all_commits()
            if not clean:
                return commit_list
            else:
                commits_dict = {}
                for commit in commit_list:
                    commits_dict[commit["commit_id"]] = commit
                return commits_dict
            
        elif type=="issue":
                artifact = Issues()
                for index, row in df.iterrows():
                    artifact.add_issue(number=row["issue_id"], body=row["issue_desc"], comments=row["issue_comments"], 
                                        createdAt=row["created_at"], updatedAt=row["closed_at"])
                issue_list = artifact.get_all_issues()
                if not clean:
                    return issue_list
                else:
                    issues_dict = {}
                    for issue in issue_list:
                        issues_dict[issue["issue_id"]] = issue
                    return issues_dict
        elif type == "link":
            if not clean:
                return df
            artifact = []
            for index, row in df.iterrows():
                artifact.append((row["issue_id"], row["commit_id"]))
            return artifact


    def read_artifacts(self, proj_data_dir):
        issues = Issues()
        commits = Commits()
        links = pd.DataFrame()
        commit_file = os.path.join(proj_data_dir, "commit.csv")
        issue_file = os.path.join(proj_data_dir, "issue.csv")
        
        issues = self.read_OSS_artifacts(issue_file, type="issue", artifact=issues)
        commits = self.read_OSS_artifacts(commit_file, type="commit", artifact=commits)
        
        if self.training:
            link_file = os.path.join(proj_data_dir, "link.csv")
            links = self.read_OSS_artifacts(link_file, type="link")
            
        return issues, commits, links

    def clean_artifacts(self, proj_dir):
        issue, commit, link = self.read_artifacts(proj_dir)
        clean_issue_file = os.path.join(proj_dir, "clean_issue.csv")
        clean_commit_file = os.path.join(proj_dir, "clean_commit.csv") 
        clean_issues = dict()
        clean_commits = dict()

        if not os.path.isfile(clean_issue_file):
            for iss in tqdm(issue, desc="Cleaning issues"):
                if pd.isnull(iss["issue_desc"]):
                    iss["issue_desc"] = ""
                iss["issue_desc"] = re.sub("<!-.*->", "", iss["issue_desc"])
                iss["issue_desc"] = re.sub("```.*```", "", iss["issue_desc"], flags=re.DOTALL)
                iss["issue_desc"] = " ".join(word_tokenize(iss["issue_desc"]))
                iss["issue_comments"] = " ".join(word_tokenize(iss["issue_comments"]))  # use only the first comment (title)
        
                clean_issues[iss["issue_id"]] = iss
        else:
            tmp_issues = self.read_OSS_artifacts(clean_issue_file, type="issue")
            for iss in tmp_issues:
                clean_issues[iss["issue_id"]] = iss

        if not os.path.isfile(clean_commit_file):
            for cm in tqdm(commit, desc="Cleaning commits"):
                diff_added_sents = eval(cm["diff_added"])
                diff_removed_sents = eval(cm["diff_removed"])
                diff_added_tokens = []
                diff_removed_tokens = []
                for sent in diff_added_sents:
                    sent = sent.strip("+ ")
                    diff_added_tokens.extend(word_tokenize(sent))
                for sent in diff_removed_sents:
                    sent = sent.strip("- ")
                    diff_removed_tokens.extend(word_tokenize(sent))
                cm["diff_added"] = emoji.replace_emoji(cm["diff_added"], "")
                cm["diff_removed"] = emoji.replace_emoji(cm["diff_removed"], "")
                cm["diff_summary"] = emoji.replace_emoji(cm["summary"], "")
                cm["diff_added"] = " ".join(diff_added_tokens)
                cm["diff_removed"] = " ".join(diff_removed_tokens)
                cm["summary"] = " ".join(word_tokenize(cm["summary"]))
                clean_commits[cm["commit_id"]] = cm
        else:
            tmp_commit = self.read_OSS_artifacts(clean_commit_file, type="commit")
            for cm in tmp_commit:
                clean_commits[cm["commit_id"]] = cm

        # save clean artifacts
        self.__save_artifacts(clean_issues.values(), output_file=clean_issue_file)
        self.__save_artifacts(clean_commits.values(), output_file=clean_commit_file)

        if not self.training:
            return
        
        logger = logging.getLogger()
        logger.setLevel("INFO")
        logger.info("Extracting links and filtering dataset")

        clean_link_file = os.path.join(proj_dir, "clean_link.csv")
        clean_links = []
        for lk in link.iterrows():
            if lk[1][0] not in clean_issues.keys() or lk[1][1] not in clean_commits.keys():
                continue
            clean_links.append((lk[1][0], lk[1][1]))
        clean_links_df = pd.DataFrame(clean_links, columns=["issue_id", "commit_id"])

        source_ids = set([x[0] for x in clean_links])
        target_ids = set([x[1] for x in clean_links])
        
        # remove artifacts do not have associated links
        remove_source = [x for x in clean_issues.keys() if x not in source_ids]
        remove_target = [x for x in clean_commits.keys() if x not in target_ids]
        for rs in remove_source:
            del clean_issues[rs]
        for rt in remove_target:
            del clean_commits[rt]

        # save clean artifacts
        self.__save_artifacts(clean_issues.values(), output_file=clean_issue_file)
        self.__save_artifacts(clean_commits.values(), output_file=clean_commit_file)
        self.__save_artifacts(clean_links_df, output_file=clean_link_file)        

    def __write_split_chunk(self, issue, commit, links, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        commit_file = os.path.join(output_dir, "commit_file")
        issue_file = os.path.join(output_dir, "issue_file")
        link_file = os.path.join(output_dir, "link_file")

        sel_issues, sel_commits = [], []
        for iss_id, cm_id in links:
            cm = commit[cm_id]
            iss = issue[iss_id]
            sel_commits.append(cm)
            sel_issues.append(iss)
        links = pd.DataFrame(links, columns=["issue_id", "commit_id"])
        self.__save_artifacts(sel_issues, output_file=issue_file)
        self.__save_artifacts(sel_commits, output_file=commit_file)
        self.__save_artifacts(links, output_file=link_file)

    def split(self, issue, commit, links, proj_dir):
        train_dir = os.path.join(proj_dir, "train")
        valid_dir = os.path.join(proj_dir, "valid")
        test_dir = os.path.join(proj_dir, "test")

        random.shuffle(links)
        train_pop = int(len(links) * 0.8)
        valid_pop = int(len(links) * 0.1)
        test_pop = int(len(links) * 0.1)

        train_links = links[:train_pop]
        valid_links = links[train_pop: train_pop + valid_pop]
        test_links = links[-test_pop:]

        self.__write_split_chunk(issue, commit, train_links, train_dir)
        self.__write_split_chunk(issue, commit, valid_links, valid_dir)
        self.__write_split_chunk(issue, commit, test_links, test_dir)

def main(args):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel("INFO")

    config = configparser.ConfigParser()
    config.read("credentials.cfg")

    proj_data_dir = os.path.join(args.root_data_dir, args.repo_path)
        
    if not (os.path.exists(os.path.join(proj_data_dir, "clean_issue.csv")) 
            and os.path.exists(os.path.join(proj_data_dir, "clean_commit.csv"))):
        # if the issue_csv is not available
        logger.info("Processing repo: {}".format(args.repo_path))
        git_token = config["github"]["token"]
        download_dir = "../../../git_repos"
        rpc = GitRepoCollector(git_token, download_dir, args.root_data_dir, 
                               args.repo_path, training=args.training)
        rpc.create_issue_commit_dataset()

        data_processing = DataProcess(args.training)

        data_processing.clean_artifacts(proj_data_dir)
    
    logger.info("Cleaned issues and commits are stored")

    if not args.training:
        return

    # Create training splits if needed
    clean_issue_file = os.path.join(proj_data_dir, "clean_issue.csv")
    clean_commits_file = os.path.join(proj_data_dir, "clean_commit.csv")
    clean_links_file = os.path.join(proj_data_dir, "clean_link.csv")
    
    data_processing = DataProcess(args.training)
    clean_issues = data_processing.read_OSS_artifacts(clean_issue_file, "issue", clean=True)
    clean_commits = data_processing.read_OSS_artifacts(clean_commits_file, "commit", clean=True)
    clean_links = data_processing.read_OSS_artifacts(clean_links_file, "link", clean=True)
    
    data_processing.split(clean_issues, clean_commits, clean_links, proj_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", default="scikit-learn/scikit-learn")
    parser.add_argument("--root_data_dir", default="../../data/git_data/training related data v2")
    parser.add_argument("--training", default=True)

    main(parser.parse_args())