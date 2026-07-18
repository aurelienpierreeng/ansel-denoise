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
| `bundle_sha256` | hash of the ingested .tar.gz |
| `n_shards_new` | shards actually merged (duplicates are skipped) |
| `n_tiles` / `cameras` | corpus statistics of the bundle |
| `license` | `ATDL-1.0` — the Ansel Training Data License ([LICENSE-DATA.md](../LICENSE-DATA.md)); the signed grant is in the bundle manifest |

`pending/` is the submission queue: each contribution pull request (opened by
`scripts/submit_contribution.sh`) adds one JSON metadata file there — handle,
download link, bundle sha256, statistics, license grant. The maintainer
ingests it with `scripts/collect_contribution.sh contrib/pending/<file>.json`
and removes the pending file in the same commit that adds the registry entry.

To remove a contribution later: delete the `<handle>_*.npz` names from the
`shards-v1` release assets and from `published.txt`, and append a
tombstone line here rather than rewriting history.
