#!/usr/bin/env python3
"""
Script to check the status of GitHub Actions workflows, specifically the documentation deployment.
This script uses the GitHub API to determine if the docs workflow has run successfully
and if the GitHub Pages documentation is available.
"""

import requests
import json
from datetime import datetime

def check_github_actions_status(owner, repo):
    """
    Check the status of GitHub Actions workflows for a repository.
    
    Args:
        owner: Repository owner
        repo: Repository name
    
    Returns:
        dict: Workflow status information
    """
    # GitHub API endpoint for workflow runs
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    
    # Make API request
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Could not fetch workflow runs. Status code: {response.status_code}")
        return None
    
    workflow_runs = response.json()
    
    # Find the documentation deployment workflow
    docs_workflows = []
    for run in workflow_runs.get('workflow_runs', []):
        if run['name'] == 'Deploy Documentation' or 'docs' in run['name'].lower():
            docs_workflows.append(run)
    
    if docs_workflows:
        # Print most recent run
        latest_run = docs_workflows[0]
        print(f"Found documentation workflow: '{latest_run['name']}'")
        print(f"Status: {latest_run['status']}")
        print(f"Conclusion: {latest_run['conclusion']}")
        print(f"Started at: {latest_run['created_at']}")
        print(f"URL: {latest_run['html_url']}")
        
        # Check if it was successful
        if latest_run['status'] == 'completed' and latest_run['conclusion'] == 'success':
            print("\n✅ Documentation workflow completed successfully!")
            return True
        else:
            print(f"\n❌ Documentation workflow did not complete successfully. Last status: {latest_run['conclusion']}")
            return False
    else:
        print("No documentation deployment workflow found.")
        # Print all workflows to see what's available
        all_workflows = []
        for run in workflow_runs.get('workflow_runs', []):
            all_workflows.append(run['name'])
        unique_workflows = list(set(all_workflows))
        print(f"Available workflows: {unique_workflows}")
        return None

def check_github_pages(owner, repo):
    """
    Check if GitHub Pages is enabled and the documentation is available.
    
    Args:
        owner: Repository owner
        repo: Repository name
    
    Returns:
        bool: True if GitHub Pages is accessible, False otherwise
    """
    # GitHub Pages URL
    pages_url = f"https://{owner}.github.io/{repo}/"
    
    try:
        response = requests.get(pages_url)
        if response.status_code == 200:
            print(f"✅ GitHub Pages is accessible at {pages_url}")
            return True
        else:
            print(f"❌ GitHub Pages returned status code: {response.status_code}")
            print(f"   URL: {pages_url}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing GitHub Pages: {e}")
        print(f"   URL: {pages_url}")
        return False

def main():
    print("Checking GitHub Actions workflow status for documentation deployment...")
    print("="*60)
    
    # Repository information
    owner = "edithatogo"
    repo = "voiage"
    
    # Check GitHub Actions workflow status
    workflow_status = check_github_actions_status(owner, repo)
    
    print("\n" + "="*60)
    print("Checking GitHub Pages accessibility...")
    
    # Check if GitHub Pages is accessible
    pages_accessible = check_github_pages(owner, repo)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    if workflow_status == True:
        print("✅ Documentation workflow has completed successfully")
    elif workflow_status == False:
        print("❌ Documentation workflow did not complete successfully")
    else:
        print("❓ Documentation workflow not found or status unclear")
    
    if pages_accessible:
        print("✅ GitHub Pages documentation is accessible")
    else:
        print("❌ GitHub Pages documentation is not accessible")
    
    print("\nIf the documentation workflow has not yet run or failed,")
    print("the GitHub Pages site will not be available.")
    print("\nTo fix: Ensure the docs workflow runs successfully by:")
    print("1. Making sure the docs.yml workflow file is properly configured")
    print("2. Ensuring there are no errors in the Sphinx documentation build")
    print("3. Confirming GitHub Pages is enabled in the repository settings")

if __name__ == "__main__":
    main()