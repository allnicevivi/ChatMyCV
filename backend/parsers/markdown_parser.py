"""
parsing content from markdown file
"""

#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")

from utils.app_logger import LoggerSetup
from component.base import Node

from typing import Any, List, Tuple, Optional, Dict
import re
from pathlib import Path
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
import pprint as pp

logger = LoggerSetup("MDParser").logger


from typing import Any, List, Tuple, Optional, Dict
import re
from pathlib import Path
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from component.base import Node

class MarkdownReader():
    """Markdown parser.

    Extract text from markdown files.
    Returns dictionary with keys as headers and values as the text between headers.
    """

    def __init__(
        self,
        *args: Any,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to a dictionary.

        The keys are the headers and the values are the text under each header.

        """
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")

        current_header = None
        current_lines = []
        in_code_block = False

        current_layer = 0
        for line in lines:
            if line.startswith("```"):
                # This is the end of a code block if we are already in it, and vice versa.
                in_code_block = not in_code_block

            header_match = re.match(r"^#+\s", line)
            if not header_match:
                header_match = re.match(r"^#+", line)
            if not in_code_block and header_match:
                # Upon first header, skip if current text chunk is empty
                if current_header is not None or len(current_lines) > 0:
                    # print(f'current_layer: {current_layer}, current_header: {current_header}')
                    markdown_tups.append((current_layer, current_header, "\n".join(current_lines)))

                current_header = line
                current_layer = len(header_match.group().strip())
                current_lines.clear()
            else:
                current_lines.append(line)
            
        # Append final text chunk
        markdown_tups.append((current_layer, current_header, "\n".join(current_lines)))

        # Postprocess the tuples before returning
        return [
            (
                layer,
                key if key is None else re.sub(r"#", "", key).strip(),
                re.sub(r"<.*?>", "", value),
            )
            for layer, key, value in markdown_tups
        ]

    def remove_images(self, content: str) -> str:
        """Remove images in markdown content."""
        pattern = r"!{1}\[\[(.*)\]\]"
        return re.sub(pattern, "", content)

    def remove_hyperlinks(self, content: str) -> str:
        """Remove hyperlinks in markdown content."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        return re.sub(pattern, r"\1", content)

    def parse_tups(
        self,
        filepath: Path,
        content: str = "",
        errors: str = "ignore",
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Tuple[Optional[str], str]]:
        """Parse file into tuples."""
        if content:
            return self.markdown_to_tups(content)
        
        fs = fs or LocalFileSystem()
        with fs.open(filepath, encoding="utf-8") as f:
            content = f.read().decode(encoding="utf-8")
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        return self.markdown_to_tups(content)

    def load_data(
        self,
        file: Path,
        content: str="",
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Node]:
        """Parse file into string."""
        tups = self.parse_tups(file, content, fs=fs)
        results = []
        if not extra_info: extra_info = {}
        extra_info["filename"] = file.name
        # TODO: don't include headers right now
        for layer, header, value in tups:
            if header is None:
                # extra_info["is_gpt_converted"] = True
                results.append(Node(text=value, metadata=extra_info or {}))
            else:
                extra_info[f'Header_{layer}'] = header
                i = 1
                while extra_info.get(f'Header_{layer+i}', None):
                    extra_info.pop(f'Header_{layer+i}')
                    i += 1
                if not value:
                    continue
                    
                headers = '\n'.join([v for k, v in extra_info.items() if k.startswith("Header_")])
                results.append(
                    Node(text=f"{headers}\n{value}", metadata=extra_info or {})
                )
        return results


if __name__ == "__main__":
    md_file = Path("./data/en/2025.11_ElgoAI.md")

    documents = MarkdownReader().load_data(
        file=md_file,
    )

    pp.pprint(documents)
