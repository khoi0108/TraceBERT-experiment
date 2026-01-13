import requests
import configparser
# GitHub GraphQL API URL
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

# Query to get github issues
GITHUB_GRAPHQL_QUERY = """
query($issuesCursor: String, $owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    basicIssues: issues(first: 50, after: $issuesCursor) { 
    	pageInfo {
        endCursor
        hasNextPage
      }
      edges {
        node {
          number
          title
          body
          createdAt
          updatedAt
          comments(first: 100) {
            edges {
              node {
                id
                bodyText
              }
            }
          }
        }
      }
    }
  }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}

"""
TEST_REPO_PATH = "bytedance/UI-TARS-desktop"

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
                json={"query": GITHUB_GRAPHQL_QUERY, "variables": variables}
            )
            if response.status_code == 200:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making GraphQL request: {e}")
            return None
        

def get_issues(repo_path, token):
        """
				Executes a GraphQL query to fetch all issues and comments from a GitHub repository.

				Args:
						repo_path (str): The path to the GitHub repository.
						token (str): The GitHub token for authentication.

				Returns:
						tuple: A tuple containing lists of all issues.
				"""
        owner, name = repo_path.split('/')

        variables = {
            "issuesCursor": None,
            "owner": owner,
            "name": name
        }

        all_issues = []

        # Make the GraphQL request using the imported function
        while True:
            hasNextPage = False
            response_data = make_github_graphql_request(token, variables)
            if response_data:
                try:

                    basic_issues = response_data["data"]["repository"]["basicIssues"]

                    # Append issues and pull requests data
                    all_issues.extend(basic_issues["edges"])

                    # Check for pagination
                    if basic_issues["pageInfo"]["hasNextPage"]:
                        variables["issuesCursor"] = basic_issues["pageInfo"]["endCursor"]
                        hasNextPage = True

                    if not hasNextPage:
                        break
                except:
                    print('Error occured while executing query')
                    break
            else:
                print("GraphQL request failed, stopping further processing.")
                break
        return all_issues


if __name__ == "__main__":
    
    config = configparser.ConfigParser()
    config.read("credentials.cfg")
    token = config.get("github", "token")
    issues = get_issues(TEST_REPO_PATH, token)
    print(issues)
