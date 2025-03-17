from git_parser import GitRepoFetcher

def main():
    """Main function to fetch repositories and process them."""
    print("Starting repository cloning and processing...")
    fetcher = GitRepoFetcher()
    fetcher.fetch_repos()
    print("Processing completed.")

if __name__ == "__main__":
    main()
