"""Offline tests of the PlayRaw crawler's parsing logic."""

from ansel_denoise.crawl_playraw import detect_license, extract_raw_links


def test_detect_license_from_cc_link():
    assert detect_license('<a href="https://creativecommons.org/licenses/by-sa/4.0/">x</a>') == "by-sa"
    assert detect_license('<a href="https://creativecommons.org/licenses/by/4.0/">x</a>') == "by"
    assert detect_license('<a href="https://creativecommons.org/publicdomain/zero/1.0/">x</a>') == "cc0"


def test_detect_license_from_text():
    assert detect_license("<p>These files are licensed CC0.</p>") == "cc0"
    assert detect_license("<p>This file is licensed Creative Commons, By-Attribution, Share-Alike.</p>") == "by-sa"
    assert detect_license("<p>Creative Commons, By-Attribution, Non-Commercial</p>") == "by-nc"
    assert detect_license("<p>just a photo, no license here</p>") is None


def test_extract_raw_links():
    cooked = '''
      <a href="/uploads/short-url/rWGtb8B4nFWPae6j5G2uPkRSvAj.ORF">shot.ORF</a>
      <a href="/uploads/short-url/5QGw8hC1B0eN7W31ryOoh45OJcG.xmp">edit.xmp</a>
      <a href="/uploads/short-url/bFwpYLw5uRwk6S00z6yQXpbf6P7.jpeg?dl=1">preview</a>
      <a href="https://example.com/other.NEF">external</a>
      <a href="https://example.com/other.NEF">duplicate</a>
      <a href="https://example.com/movie.x3f">foveon</a>
    '''
    links = extract_raw_links(cooked, "https://discuss.pixls.us")
    assert links == [
        "https://discuss.pixls.us/uploads/short-url/rWGtb8B4nFWPae6j5G2uPkRSvAj.ORF",
        "https://example.com/other.NEF",
    ]
