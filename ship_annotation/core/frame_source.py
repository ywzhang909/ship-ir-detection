import os
from pathlib import Path
from typing import List, Optional
from .data_models import SourceType


class FrameSource:
    """图像源管理"""

    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self):
        self._source_type = SourceType.IMAGE
        self._current_path: Optional[str] = None
        self._folder_path: Optional[str] = None
        self._image_list: List[str] = []
        self._current_index: int = 0

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    @property
    def image_list(self) -> List[str]:
        return self._image_list

    @property
    def current_index(self) -> int:
        return self._current_index

    def open_folder(self, folder_path: str) -> List[str]:
        """打开文件夹，返回图片列表"""
        self._folder_path = folder_path
        self._source_type = SourceType.IMAGE
        self._image_list = []

        for f in sorted(os.listdir(folder_path)):
            if Path(f).suffix.lower() in self.SUPPORTED_EXTS:
                self._image_list.append(os.path.join(folder_path, f))

        self._current_index = 0
        if self._image_list:
            self._current_path = self._image_list[0]

        return self._image_list

    def select_image(self, path: str):
        """选择单张图片"""
        self._current_path = path
        if path in self._image_list:
            self._current_index = self._image_list.index(path)

    def next_image(self) -> Optional[str]:
        """下一张图片"""
        if not self._image_list:
            return None
        self._current_index = min(self._current_index + 1, len(self._image_list) - 1)
        self._current_path = self._image_list[self._current_index]
        return self._current_path

    def prev_image(self) -> Optional[str]:
        """上一张图片"""
        if not self._image_list:
            return None
        self._current_index = max(self._current_index - 1, 0)
        self._current_path = self._image_list[self._current_index]
        return self._current_path
