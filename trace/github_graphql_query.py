import requests
# GitHub GraphQL API URL
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

GITHUB_GRAPHQL_QUERY = """
query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      issues {
        totalCount
      }
    }
}
"""

def make_github_graphql_request(token, variables):
        """
        Makes a GraphQL request to GitHub API.

        Args:
        - token: GitHub personal access token
        - query: GraphQL query string
        - variables: Variables for the GraphQL query

        Returns:
        - JSON response from GitHub API
        """

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                GITHUB_GRAPHQL_URL,
                headers=headers,
                json={"query":GITHUB_GRAPHQL_QUERY, "variables": variables}
            )
            if response.status_code == 200:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making GraphQL request: {e}")
            return None
        

def get_issue_count(repo_path, token):
        """
				Executes a GraphQL query to fetch issues, pull requests, and issue links from a GitHub repository.

				Args:
						repo_path (str): The path to the GitHub repository.
						token (str): The GitHub token for authentication.

				Returns:
						tuple: A tuple containing lists of all issues, all pull requests, and all issue links.
				"""
        owner, name = repo_path.split('/')

        variables = {
            "owner": owner,
            "name": name
        }

        issue_count = 0

        # Make the GraphQL request using the imported function
        while True:
            response_data = make_github_graphql_request(token, variables)
            if response_data:
                try:
                    issue_count = response_data['data']['repository']['issues']['totalCount']
                    return issue_count
                except:
                    print('Error occured while executing query')
                    break
            else:
                print("GraphQL request failed, stopping further processing.")
                break