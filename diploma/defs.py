from typing import Literal
TARGET_SR = 16000

# Dataset
STAGES = Literal['train', 'val', 'test']
LETTERS = [
    "_",
    " ",
    "а",
    "ә",
    "б",
    "в",
    "г",
    "ғ",
    "д",
    "е",
    "ё",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "қ",
    "л",
    "м",
    "н",
    "ң",
    "о",
    "ө",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ұ",
    "ү",
    "ф",
    "х",
    "һ",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "і",
    "ь",
    "э",
    "ю",
    "я",
]
CHAR2IDX = {char: idx for idx, char in enumerate(LETTERS)}
IDX2CHAR = {idx: char for char, idx in CHAR2IDX.items()}

