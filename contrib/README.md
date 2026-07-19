# Contribution registry

`registry.jsonl` is the committed bookkeeping of every community shard
contribution merged into the training corpus: one JSON line per collected
bundle, appended by `scripts/collect_contribution.sh`.

Fields per entry:

| field | meaning |
|---|---|
| `collected` | UTC timestamp when the maintainer merged the bundle |
| `handle` | contributor handle (also the filename prefix of their shards) |
| `created` | UTC timestamp when the contributor packed the bundle |
| `source` | where the bundle came from (download URL or issue link) |
| `bundle_sha256` | hash of the ingested .tar.gz as downloaded |
| `n_shards_new` | shards actually merged (duplicates are skipped) |
| `n_tiles` / `cameras` | corpus statistics of the bundle |
| `license` | the ATDL version granted at pack time, e.g. `ATDL-1.1` ([LICENSE-DATA.md](../LICENSE-DATA.md)); the signed grant is in the bundle manifest |

Submission is by GitHub issue: a contributor uploads their `.tar.gz` to a file
host and opens a 'Shard contribution' issue with the link. The maintainer
ingests it with `scripts/collect_contribution.sh <link> --source <issue-url>`,
which downloads, verifies the manifest's per-file hashes, merges the shards
and appends the registry line — then commits `registry.jsonl`.

To remove a contribution later: delete the `<handle>_*.npz` names from the
`shards-v1` release assets and from `published.txt`, and append a
tombstone line here rather than rewriting history.
