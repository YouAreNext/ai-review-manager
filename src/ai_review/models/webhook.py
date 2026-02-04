from pydantic import BaseModel


class GitLabUser(BaseModel):
    username: str
    name: str | None = None


class GitLabProject(BaseModel):
    id: int
    path_with_namespace: str
    web_url: str


class GitLabMergeRequest(BaseModel):
    iid: int
    title: str
    source_branch: str
    target_branch: str
    state: str
    action: str | None = None


class GitLabMREvent(BaseModel):
    object_kind: str  # "merge_request"
    user: GitLabUser
    project: GitLabProject
    object_attributes: GitLabMergeRequest


class GitLabNote(BaseModel):
    note: str
    noteable_type: str


class GitLabNoteEvent(BaseModel):
    object_kind: str  # "note"
    user: GitLabUser
    project: GitLabProject
    merge_request: GitLabMergeRequest | None = None
    object_attributes: GitLabNote
