"""
submit.py — GitHub OAuth flow and Pull Request creation/update.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["submit"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_URL = "https://api.github.com"
BRANCH_NAME = "challenge-submission"


# ── OAuth flow ──────────────────────────────────────────────────────────────

@router.get("/auth/github")
def github_login():
    client_id = os.environ["GITHUB_CLIENT_ID"]
    return RedirectResponse(
        f"{GITHUB_AUTHORIZE_URL}?client_id={client_id}&scope=repo"
    )


@router.get("/auth/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    error: str | None = None,
):
    frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")

    # User cancelled or GitHub returned an error — redirect back to login gate
    if error or not code:
        return RedirectResponse(f"{frontend_origin}/?auth_error=cancelled")

    client_id = os.environ["GITHUB_CLIENT_ID"]
    client_secret = os.environ["GITHUB_CLIENT_SECRET"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
            },
        )

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        return RedirectResponse(f"{frontend_origin}/?auth_error=token_failed")

    request.session["github_token"] = access_token

    # Fetch user info and store in session
    user = await _github_get(access_token, "/user")
    request.session["github_user"] = {
        "login": user["login"],
        "avatar_url": user.get("avatar_url", ""),
        "name": user.get("name", ""),
    }

    frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
    return RedirectResponse(f"{frontend_origin}/")


@router.get("/auth/me")
async def auth_me(request: Request):
    user = request.session.get("github_user")
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, "user": user}


@router.get("/auth/logout")
def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


# ── Deadline ────────────────────────────────────────────────────────────────

@router.get("/deadline")
def get_deadline():
    raw = os.environ.get("DEADLINE", "")
    if not raw:
        return {"deadline": None, "has_passed": False, "seconds_remaining": None}

    try:
        deadline_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return {"deadline": raw, "has_passed": False, "seconds_remaining": None}

    now = datetime.now(timezone.utc)
    diff = (deadline_dt - now).total_seconds()
    return {
        "deadline": deadline_dt.isoformat(),
        "has_passed": diff <= 0,
        "seconds_remaining": max(0, int(diff)),
    }


# ── Submission ──────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    pipeline_config: dict
    model_b64: str          # base64-encoded model.pkl bytes


@router.post("/submit")
async def submit(request: Request, payload: SubmitRequest):
    # ── Deadline check ──────────────────────────────────────────────────────
    deadline_info = get_deadline()
    if deadline_info.get("has_passed"):
        raise HTTPException(status_code=403, detail="Submission deadline has passed")

    # ── Auth check ──────────────────────────────────────────────────────────
    token = request.session.get("github_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated with GitHub")

    user = request.session.get("github_user", {})
    username = user.get("login")
    if not username:
        raise HTTPException(status_code=401, detail="GitHub user not found in session")

    target_repo = os.environ["TARGET_REPO"]    # e.g. "org/machine-learning-lab-2026"
    if target_repo.startswith("your-org"):
        raise HTTPException(status_code=422, detail="TARGET_REPO is not configured — update your .env file")
    target_owner, target_repo_name = target_repo.split("/", 1)

    # ── Validate model bytes ────────────────────────────────────────────────
    try:
        model_bytes = base64.b64decode(payload.model_b64)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid base64 model data")

    # ── Find or create the student's fork ──────────────────────────────────
    try:
        fork_info = await _get_or_create_fork(token, target_owner, target_repo_name)
        fork_owner = fork_info["owner"]["login"]
        fork_repo_name = fork_info["name"]

        # ── Ensure challenge-submission branch exists on fork ──────────────────
        default_branch = fork_info.get("default_branch", "main")
        branch_exists = await _branch_exists(token, fork_owner, fork_repo_name, BRANCH_NAME)

        if not branch_exists:
            sha = await _get_branch_sha(token, fork_owner, fork_repo_name, default_branch)
            await _create_branch(token, fork_owner, fork_repo_name, BRANCH_NAME, sha)

        # ── Commit files ────────────────────────────────────────────────────────
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        commit_message = f"challenge submission — {timestamp}"

        config_b64 = base64.b64encode(
            json.dumps(payload.pipeline_config, indent=2).encode()
        ).decode()
        model_b64_clean = base64.b64encode(model_bytes).decode()

        submissions_path = f"challenge/submissions/{username}"

        await _upsert_file(
            token, fork_owner, fork_repo_name, BRANCH_NAME,
            f"{submissions_path}/pipeline_config.json",
            config_b64, commit_message,
        )
        await _upsert_file(
            token, fork_owner, fork_repo_name, BRANCH_NAME,
            f"{submissions_path}/model.pkl",
            model_b64_clean, commit_message,
        )

        # ── Open or update PR ───────────────────────────────────────────────────
        pr_url = await _ensure_pull_request(
            token,
            head_owner=fork_owner,
            head_repo=fork_repo_name,
            head_branch=BRANCH_NAME,
            base_owner=target_owner,
            base_repo=target_repo_name,
            username=username,
            timestamp=timestamp,
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        url = str(exc.request.url)
        if status == 404:
            raise HTTPException(
                status_code=502,
                detail=f"Repository not found: {target_owner}/{target_repo_name}. Check TARGET_REPO in your .env file.",
            )
        if status == 403 and "forks" in url:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"The app does not have access to the '{target_owner}' organization. "
                    "Please re-authorize and click \"Grant\" next to the organization name."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error {status} on {url}. Check your token scopes and repository access.",
        )

    return {"ok": True, "pr_url": pr_url, "submitted_at": timestamp}


# ── GitHub API helpers ──────────────────────────────────────────────────────

async def _github_get(token: str, path: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GITHUB_API_URL}{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
    resp.raise_for_status()
    return resp.json()


async def _github_post(token: str, path: str, body: dict) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GITHUB_API_URL}{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json=body,
        )
    resp.raise_for_status()
    return resp.json()


async def _github_put(token: str, path: str, body: dict) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{GITHUB_API_URL}{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json=body,
        )
    resp.raise_for_status()
    return resp.json()


async def _get_or_create_fork(token: str, owner: str, repo: str) -> dict:
    # Check if fork already exists
    user_info = await _github_get(token, "/user")
    username = user_info["login"]
    try:
        fork = await _github_get(token, f"/repos/{username}/{repo}")
        if fork.get("fork") and fork["parent"]["full_name"] == f"{owner}/{repo}":
            return fork
    except httpx.HTTPStatusError:
        pass
    # Create fork
    fork = await _github_post(token, f"/repos/{owner}/{repo}/forks", {})
    return fork


async def _branch_exists(token: str, owner: str, repo: str, branch: str) -> bool:
    try:
        await _github_get(token, f"/repos/{owner}/{repo}/branches/{branch}")
        return True
    except httpx.HTTPStatusError:
        return False


async def _get_branch_sha(token: str, owner: str, repo: str, branch: str) -> str:
    data = await _github_get(token, f"/repos/{owner}/{repo}/branches/{branch}")
    return data["commit"]["sha"]


async def _create_branch(token: str, owner: str, repo: str, branch: str, sha: str):
    await _github_post(token, f"/repos/{owner}/{repo}/git/refs", {
        "ref": f"refs/heads/{branch}",
        "sha": sha,
    })


async def _get_file_sha(token: str, owner: str, repo: str, branch: str, path: str) -> str | None:
    try:
        data = await _github_get(token, f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
        return data.get("sha")
    except httpx.HTTPStatusError:
        return None


async def _upsert_file(
    token: str, owner: str, repo: str, branch: str,
    path: str, content_b64: str, message: str,
):
    existing_sha = await _get_file_sha(token, owner, repo, branch, path)
    body: dict = {
        "message": message,
        "content": content_b64,
        "branch": branch,
    }
    if existing_sha:
        body["sha"] = existing_sha
    await _github_put(token, f"/repos/{owner}/{repo}/contents/{path}", body)


async def _ensure_pull_request(
    token: str,
    head_owner: str, head_repo: str, head_branch: str,
    base_owner: str, base_repo: str,
    username: str, timestamp: str,
) -> str:
    # Check for existing PR
    head_label = f"{head_owner}:{head_branch}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GITHUB_API_URL}/repos/{base_owner}/{base_repo}/pulls",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            params={"head": head_label, "state": "open"},
        )
    resp.raise_for_status()
    prs = resp.json()

    pr_body = (
        f"**Student:** {username}\n"
        f"**Last updated:** {timestamp}\n\n"
        "This PR was automatically created by the challenge web app.\n"
        "Each new submission adds a commit to this branch — the latest commit counts."
    )

    if prs:
        # Update existing PR body
        pr = prs[0]
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{GITHUB_API_URL}/repos/{base_owner}/{base_repo}/pulls/{pr['number']}",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                json={"body": pr_body},
            )
        return pr["html_url"]

    # Create new PR
    new_pr = await _github_post(token, f"/repos/{base_owner}/{base_repo}/pulls", {
        "title": f"[Challenge] Submission — {username}",
        "head": head_label,
        "base": "main",
        "body": pr_body,
    })
    return new_pr["html_url"]
