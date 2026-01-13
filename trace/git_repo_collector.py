"""
Collect github projects by programming language
extract trace links between commits and issues
create doc string to source code relationship
"""
import calendar
import logging
import os
import time

from github import Github, \
    RateLimitExceededException, \
    Auth  # pip install PyGithub. Lib operates on remote github to get issues
from  github_graphql_query import get_issues
import re
import argparse
import git as local_git  # pip install GitPython. Lib operates on local repo to get commits
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class Issue:
    def __init__(self, issue_id: str, desc: str, comments: str, create_time, close_time):
        self.issue_id = issue_id
        self.desc = "" if pd.isnull(desc) else desc
        # self.desc = desc
        self.comments = comments
        self.create_time = create_time
        self.close_time = close_time

    def to_dict(self):
        return {
            "issue_id": self.issue_id,
            "issue_desc": self.desc,
            "issue_comments": self.comments,
            "closed_at": self.create_time,
            "created_at": self.close_time
        }

    def __str__(self):
        return str(self.to_dict())


class Commit:
    def __init__(self, commit_id, summary, diffs, files, commit_time):
        self.commit_id = commit_id
        self.summary = summary
        self.diffs = diffs
        self.files = files
        self.commit_time = commit_time

    def to_dict(self):
        return {
            "commit_id": self.commit_id,
            "summary": self.summary,
            "diff": self.diffs,
            "files": self.files,
            "commit_time": self.commit_time
        }

    def __str__(self):
        return str(self.to_dict())

    # def __str__(self):
    #     summary = re.sub("[,\r\n]+", " ", self.summary)
    #     diffs = " ".join(self.diffs)
    #     diffs = re.sub("[,\r\n]+", " ", diffs)
    #     return "{},{},{},{}\n".format(self.commit_id, summary, diffs, self.commit_time)


class GitRepoCollector:
    def __init__(self, token, download_path, output_dir, repo_path):
        self.token = token
        self.download_path = download_path
        self.repo_path = repo_path
        self.output_dir = output_dir

    def clone_project(self):
        repo_url = "https://github.com/{}.git".format(self.repo_path)
        clone_path = os.path.join(self.download_path, self.repo_path)
        if not os.path.exists(clone_path):
            logger.info("Clone {}...".format(self.repo_path))
            local_git.Repo.clone_from(repo_url, clone_path)
            logger.info("finished cloning project")
        else:
            logger.info("Skip clone project as it already exist...")
        local_repo = local_git.Repo(clone_path)
        return local_repo

    
    def wait_for_rate_limit(self, git):
        # NEW: use .resources.core instead of .core
        rate_limit = git.get_rate_limit()
        remaining = rate_limit.resources.core.remaining
        logger.info("Remaining requests = {}".format(remaining))

        while remaining < 10:
            core_rate_limit = git.get_rate_limit().resources.core
            reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
            sleep_time = reset_timestamp - calendar.timegm(time.gmtime())
            logger.info("Wait until git core API rate limit reset, reset time = {} seconds".format(sleep_time))
            for _ in tqdm(range(max(0, sleep_time)), desc="Rate Limit Wait", position=0, leave=True):
                time.sleep(1)
            remaining = git.get_rate_limit().resources.core.remaining


    def get_issues(self, issue_file_path):
        if os.path.isfile(issue_file_path) and os.path.getsize(issue_file_path) > 0:
            issue_df = pd.read_csv(issue_file_path)
        else:
            issue_df = pd.DataFrame(columns=["issue_id", "issue_desc", "issue_comments", "closed_at", "created_at"])
        start_index = issue_df.shape[0]
        git = Github(auth=Auth.Token(self.token))
        git.get_user()
        self.wait_for_rate_limit(git)
        logger.info("creating issue.csv")
        issues = get_issues(self.repo_path, self.token)

        for i in tqdm(range(start_index, len(issues)), desc="Issues", position=0, leave=True):
            try:
                issue = issues[i]["node"]
                issue_number = issue["number"]
                comments = []
                comments.append(issue["title"])
                for comment in issue["comments"]["edges"]:
                    comments.append(comment["node"]["bodyText"]) 
                desc = ""
                if issue["body"]:
                    desc = issue["body"]
                issue_close_time = issue["updatedAt"]
                issue_create_time = issue["createdAt"]
                issue = Issue(str(issue_number), desc, "\n".join(comments), issue_create_time, issue_close_time)
                issue_df_delta = pd.DataFrame([issue.to_dict()])
                issue_df = pd.concat([issue_df, issue_df_delta], ignore_index=True)
            except RateLimitExceededException:
                self.wait_for_rate_limit(git)
        self.wait_for_rate_limit(git)
        issue_df.to_csv(issue_file_path)

    def get_commits(self, commit_file_path):
        """
        Retrieves commit data from a cloned local repository, processes the data, and stores it in a CSV file.
        If the CSV file already exists, the method logs a message and returns, skipping the creation process.

        Args:
            commit_file_path (str): The path where the resulting CSV file will be saved.

        Returns:
            None. The results are stored in a CSV file at the specified path.
        """
        local_repo = self.clone_project()
        if os.path.isfile(commit_file_path):
            print("commits already existing, skip creating...")
            return
        print("creating commit.csv...")
        commit_df = pd.DataFrame(columns=["commit_id", "summary", "diff", "files", "commit_time"])
        processed_commits = set()

        # Iterate over all branches, including remote branches
        branches = list(local_repo.branches) + [ref for ref in local_repo.remotes.origin.refs if 'HEAD' not in ref.name]
        for branch in branches:
            print(f"Processing branch: {branch.name}")
            for commit in tqdm(local_repo.iter_commits(branch.name)):
                if commit.hexsha in processed_commits:
                    continue

                id = commit.hexsha
                summary = commit.summary
                create_time = commit.committed_datetime
                differs = set()
                files = []

                import subprocess   
                cmd = [
                "git", "--no-pager",
                "diff-tree",
                "--root",
                "-r",
                "-p",
                "--no-renames",
                "--no-color",
                commit.hexsha,
                ]

                current_file = None

                with subprocess.Popen(cmd, cwd=local_repo.working_dir,
                                      stdout=subprocess.PIPE, encoding='utf-8', 
                                      bufsize=1, text=True, errors='ignore') as proc:
                    for line in proc.stdout:
                        line = line.rstrip("\n")

                        # Detect new file diff header
                        if line.startswith("diff --git"):
                            parts = line.split()
                            if len(parts) >= 4:
                                current_file = parts[2][2:]  # strip "a/"
                                files.append(current_file)

                        # Collect added/removed lines, ignore diff headers
                        elif current_file and line and line[0] in "+-" and not line.startswith(("+++", "---", "@@")):
                            differs.add(line)

                    proc.wait()

                commit_record = {
                    "commit_id": id,
                    "summary": summary,
                    "diff": list(differs),
                    "files": files,
                    "commit_time": create_time
                }
                commit_df = pd.concat([commit_df, pd.DataFrame([commit_record])], ignore_index=True)
                processed_commits.add(id)
        commit_df.to_csv(commit_file_path, index=False)

    def get_issue_commit_links(self, link_file_path, issue_file_path, commit_file_path, link_pattern='#\d+'):
        # Extract links from the commits
        if os.path.isfile(link_file_path):
            logger.info("link file already exists, skip creating...")
            return
        with open(link_file_path, 'w', encoding='utf8') as fout:
            fout.write("issue_id,commit_id\n")
            issue_df = pd.read_csv(issue_file_path)
            commit_df = pd.read_csv(commit_file_path)
            issue_ids = set([str(x) for x in issue_df['issue_id']])

            commit_ids = commit_df['commit_id']
            commit_summary = commit_df['summary']
            for commit_id, summary in zip(commit_ids, commit_summary):
                issue_id_pattern = link_pattern
                try:
                    res = re.search(issue_id_pattern, summary)
                except:
                    pass
                if res is not None:
                    linked_issue_id = res.group(0)
                    issue_id = linked_issue_id.strip("#")
                    if issue_id not in issue_ids:
                        logger.warning("{} is not in the issue file".format(issue_id))
                    else:
                        fout.write("{},{}\n".format(issue_id, commit_id))

    def create_issue_commit_dataset(self):
        output_dir = os.path.join(self.output_dir, self.repo_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        issue_file_path = os.path.join(output_dir, "issue.csv")
        commit_file_path = os.path.join(output_dir, "commit.csv")
        link_file_path = os.path.join(output_dir, "link.csv")

        if not os.path.isfile(issue_file_path):
            self.get_issues(issue_file_path)
        if not os.path.isfile(commit_file_path):
            self.get_commits(commit_file_path)
        self.get_issue_commit_links(link_file_path, issue_file_path, commit_file_path)
        return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Github script")
    parser.add_argument("-t", help="token")
    parser.add_argument("-d", help="download path")
    parser.add_argument("-o", help="output dir root")
    parser.add_argument("-r", nargs="+", help="repo path in github, a list of repo path can be passed")

    args = parser.parse_args()
    for repo_path in args.r:
        logger.info("Processing repo: {}".format(repo_path))
        rpc = GitRepoCollector(args.t, args.d, args.o, repo_path)
        rpc.create_issue_commit_dataset()
