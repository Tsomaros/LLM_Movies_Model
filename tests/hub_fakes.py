from pathlib import Path

from huggingface_hub import CommitOperationAdd


class FakeHub:
    """A private Hub repo (model or dataset) in a dict. create_commit applies all operations or none."""

    def __init__(self, private=True):
        self.private, self.files, self.commits, self.fail_commit = private, {}, [], False
        self.downloads = []

    def create_repo(self, **kw):
        pass

    def dataset_info(self, repo_id):
        return self.model_info(repo_id)

    def model_info(self, repo_id):
        return type("Info", (), {"private": self.private})()

    def repo_exists(self, repo_id, repo_type=None):
        return True

    def list_repo_files(self, repo_id, repo_type=None):
        return sorted(self.files)

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        self.downloads.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.fail_commit:
            raise RuntimeError("session killed")
        staged = {}
        for op in operations:
            assert isinstance(op, CommitOperationAdd)
            src = op.path_or_fileobj
            staged[op.path_in_repo] = src if isinstance(src, bytes) else Path(src).read_bytes()
        self.files.update(staged)
        self.commits.append([op.path_in_repo for op in operations])
