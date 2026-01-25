---
name: Production
about: Used to track production start-up in case of acceptance
title: ''
labels: ''
assignees: ''

---

## Phase 1: acceptance [authors + AE]

When the Associate Editor (AE) is satisfied with the author's answers to the reviewers comments, he/she exchanges with authors  that (via the discussion tool on OpenReview)

- [ ] authors are informed of final acceptance 
- [ ] affiliation, author-url, affiliation-url and other metadata are correctly filled in on the author git repository
- [ ] the manuscript is formatted using the latest Computo extension
- [ ] CI/github-action validates the reproducibility of the manuscript

The authors inform the AE upon completion of the tasks required on their side.
 
## Phase 2: production start-up [AE]

- [ ] the AE asks the editor a DOI for this publication
- [ ] the AE invites the corresponding author as a collaborator of computorg: https://github.com/orgs/computorg/people
- [ ] the AE asks the corresponding author to transfer the ownership of their repo to computorg: at the bottom of https://github.com/corresp-author/repo-name/settings, click  "Transfer ownership" and choose "computorg" in "Select one of my organizations"
- [ ] the AE renames this repo to `published-yearmonth-first_author_last_name-manuscript_keyword`, with CC-by 4 licence (year-month in format YYYYMM; use only hyphens, e.g. `published-202407-legrand-wildfires`) and ensures that the repo is public (will appear "in the pipeline"  on the web site if `draft: true`
- [ ] the AE makes an Issue asking for final proofreading by the authors (possibly with some minor additional questions), made as a pull-request on Computorg's repository
- [ ] the AE, assisted by the technical team, ensure that CI works 
- [ ] the AE makes a link to the reviews OR or PCI) in an Issue of the repository, [following this model](https://github.com/computorg/published-paper-tsne/issues/4) and add the label "Review" to each such issue (which will automatically link to the review badge in the HTML version of the paper)
- [ ] the AE archives the repository on [software heritage](https://archive.softwareheritage.org/save/)
- [ ] the AE completes the README.md according to [this template](https://github.com/computorg/published-paper-tsne/blob/main/README.md), with appropriate badges
- [ ] the AE informs the EiC that phase 2 has ended(check that proofread Issue is closed)

## Phase 3: final publication [EiC]

- [ ] the EiC set the metadata appropriately (draft: false, repos, doi, google-scholar: true; date: publication date under format MM-DD-YYYY; date-modified: last-modified.)
- [ ] the EiC creates a release of the firstly published version, tagged v1.0, on GitHub `git tag v1.0 -a -s -m "Publication v1.0"`
- [ ] the EiC ensures that the [publications page](https://computo-journal.org/computorg-new.github.io/site/publications.html) refreshes correctly
- [ ] the EiC ask the CM to communicate about the new publication is made
