import subprocess
import xml.etree.ElementTree as ET
import base64
import urllib.parse as urlparse
from urllib.parse import urlencode
from typing import Optional
import os


def generate_shield_url(
    svg_path: str,
    right_text: str,
    color: str,
    left_text: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate a shield badge url based on the provided SVG file and parameters.

    Parameters
    ----------
    svg_path : str
        The path to the SVG file to use as a base for the badge.

    right_text : str
        The text to display on the right side of the badge.

    color : str
        The color of the badge.

    left_text : Optional[str]
        The text to display on the left side of the badge. If None, only right_text


    Returns
    -------
    str
        The URL of the generated badge.
        When using this URL, the data will be send to the shields.io service to generate the badge.
    """

    base_url = "https://img.shields.io/badge/"
    tree = ET.parse(path)
    root = tree.getroot()

    if not root.tag.endswith("svg"):
        raise ValueError("This is not an svg file")

    text = right_text
    if left_text is not None:
        text = f"{left_text}-{text}"
    if color is not None:
        text = f"{text}-{color}"
    text = text + ".svg"

    shield_url = base_url + text
    parsed_url = list(urlparse.urlparse(shield_url))
    query = dict(urlparse.parse_qsl(parsed_url[4]))
    query.update(kwargs)

    # Add logo
    xmlstr = ET.tostring(root, encoding='utf8', method='xml')
    b64str = base64.encodebytes(xmlstr).decode("utf-8")

    base_str = f"data:image/svg+xml;base64,{b64str}"
    base_str = base_str.replace("\n", "")
    # query["logo"] = base_str
    pq = urlencode(query)
    pq = "logo=" + base_str + "&" + pq
    pq = pq.rstrip("&")
    parsed_url[4] = pq
    complete_url = urlparse.urlunparse(parsed_url)
    return complete_url


def save_shield(svg_path: str,
                right_text: str,
                color: str,
                save_path: str,
                left_text: Optional[str] = None,
                **kwargs) -> str:
    """
    Generate a shield badge and save it to the specified path.

    Parameters
    ----------
    svg_path : str
        The path to the SVG file to use as a base for the badge.

    right_text : str
        The text to display on the right side of the badge.

    color : str
        The color of the badge.

    save_path : str
        The path where the badge image will be saved.

    left_text : Optional[str]
        The text to display on the left side of the badge. If None, only right_text

    Returns
    -------
    str
        The path where the badge image was saved.
    """
    shield_url = generate_shield_url(
        svg_path=svg_path, right_text=right_text, left_text=left_text, color=color, **kwargs)
    # Download the shield with requests
    import requests
    response = requests.get(shield_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path


def generate_shield_html(
    svg_path: str,
    right_text: str,
    color: str,
    left_text: Optional[str] = None,
    link: Optional[str] = None,
    alt_text: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate an HTML anchor tag with an image badge based on the provided SVG file and parameters.

    Parameters
    ----------
    svg_path : str
        The path to the SVG file to use as a base for the badge.

    right_text : str
        The text to display on the right side of the badge.

    color : str
        The color of the badge.

    left_text : Optional[str]
        The text to display on the left side of the badge. If None, only right_text

    link : Optional[str]
        The URL to link the badge to. If None, no link will be created.

    alt_text : Optional[str]
        The alt text for the badge image. If None, no alt text will be added.

    Returns
    -------
    str
        The HTML string for the badge image wrapped in an anchor tag.
        When using this HTML, the data will be send to the shields.io service to generate the badge.
    """
    shield_url = generate_shield_url(
        svg_path=svg_path, right_text=right_text, left_text=left_text, color=color, **kwargs)
    link_text = None
    if link is not None:
        link_text = f'href="{link}"'
    alt_text = None
    if alt_text is not None:
        alt_text = f'alt="{alt_text}"'
    html = f'<a{" " + link_text if link_text is not None else ""}{" " + alt_text if alt_text is not None else ""}>\n\t<img src="{shield_url}"/>\n</a>'
    return html
