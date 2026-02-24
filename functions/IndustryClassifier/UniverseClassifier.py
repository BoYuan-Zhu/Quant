import os
import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # 当前这个 .py 文件所在目录
DEFAULT_CSV = (BASE_DIR / "data" / "universe_with_industry.csv").resolve()
MAPPING_PATH = (BASE_DIR/ "data" / "category_code_map.csv" ).resolve()

class UniverseClassifier:
    """
    Two usage modes:

    (A) Build mapping from a raw universe CSV (messy columns) then save:
        uc = UniverseClassifier(raw_csv_path)
        uc.build_and_save(mapping_csv_path)

    (B) Load directly from the saved mapping CSV:
        uc = UniverseClassifier.from_mapping_csv(mapping_csv_path)
    """

    def __init__(
        self,
        csv_path: str = str(DEFAULT_CSV),
        code_col: Optional[str] = None,
        category_col: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self._user_code_col = code_col
        self._user_category_col = category_col

        self.df: Optional[pd.DataFrame] = None  # normalized mapping: ["category","code"]

    # --------------------- factory: load from clean mapping ---------------------
    @classmethod
    def from_mapping_csv(cls, mapping_csv_path: str = str(MAPPING_PATH)) -> "UniverseClassifier":
        """
        Load a pre-built mapping CSV with columns: category, code (6-digit).
        """
        obj = cls(mapping_csv_path)
        df = pd.read_csv(mapping_csv_path, dtype=str)

        if "category" not in df.columns or "code" not in df.columns:
            raise ValueError(
                f"mapping csv must contain columns ['category','code'], got {list(df.columns)}"
            )

        df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        df = df.dropna(subset=["category", "code"])
        df = df[df["code"].str.fullmatch(r"\d{6}", na=False)]
        obj.df = df[["category", "code"]].reset_index(drop=True)
        return obj

    # --------------------- internal helpers ---------------------
    @staticmethod
    def _infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _to_6digit_code(s: pd.Series) -> pd.Series:
        return s.astype(str).str.extract(r"(\d{6})", expand=False)

    def _load_raw_and_normalize(self) -> pd.DataFrame:
        """
        Load raw csv and return normalized mapping DataFrame: ["category","code"]
        """
        df = pd.read_csv(self.csv_path, dtype=str)

        code_col = self._user_code_col or self._infer_column(
            df, ["symbol", "symbol_raw", "code", "股票代码", "证券代码", "ts_code"]
        )

        category_col = self._user_category_col or self._infer_column(
            df,
            [
                "industry_l1", "industry_l2", "industry_l3", "industry_standard",
                "industry", "type", "分类", "行业", "板块", "申万行业", "中信行业"
            ],
        )

        if code_col is None or category_col is None:
            raise ValueError(
                f"Cannot infer columns.\n"
                f"Existing columns = {list(df.columns)}\n"
                f"Please pass code_col='...' and category_col='...'."
            )

        df[code_col] = self._to_6digit_code(df[code_col])
        out = df[[category_col, code_col]].dropna()
        out = out.rename(columns={category_col: "category", code_col: "code"})
        out = out[out["code"].str.fullmatch(r"\d{6}", na=False)]
        out = out.drop_duplicates(subset=["category", "code"]).reset_index(drop=True)
        return out

    # --------------------- build & save ---------------------
    def build(self) -> "UniverseClassifier":
        """Build normalized mapping in memory from the raw csv."""
        self.df = self._load_raw_and_normalize()
        return self

    def build_and_save(self, mapping_csv_path: str = str(MAPPING_PATH)) -> str:
        """
        Build mapping from raw csv and save as clean mapping CSV (category, code).
        Returns the saved path.
        """
        mapping = self._load_raw_and_normalize()

        os.makedirs(os.path.dirname(mapping_csv_path) or ".", exist_ok=True)
        mapping.to_csv(mapping_csv_path, index=False, encoding="utf-8-sig")

        self.df = mapping
        return mapping_csv_path

    # --------------------- query APIs ---------------------
    def categories(self, sort: bool = True) -> List[str]:
        if self.df is None:
            raise RuntimeError("Mapping not loaded. Call build()/build_and_save() or from_mapping_csv().")
        cats = self.df["category"].dropna().unique().tolist()
        return sorted(cats) if sort else cats

    def codes(
        self,
        category: Union[str, List[str]],
        sort: bool = True,
        unique: bool = True,
    ) -> List[str]:
        if self.df is None:
            raise RuntimeError("Mapping not loaded. Call build()/build_and_save() or from_mapping_csv().")
        cat_list = [category] if isinstance(category, str) else list(category)

        sub = self.df[self.df["category"].isin(cat_list)]["code"].dropna()
        if unique:
            sub = sub.drop_duplicates()
        codes = sub.tolist()
        return sorted(codes) if sort else codes

    def category_map(self, sort_codes: bool = True) -> Dict[str, List[str]]:
        if self.df is None:
            raise RuntimeError("Mapping not loaded. Call build()/build_and_save() or from_mapping_csv().")
        res: Dict[str, List[str]] = {}
        for cat, g in self.df.groupby("category"):
            lst = g["code"].dropna().drop_duplicates().tolist()
            res[str(cat)] = sorted(lst) if sort_codes else lst
        return res

if __name__ == "__main__":


    uc = UniverseClassifier(code_col="symbol", category_col="industry_l3")
    uc.build_and_save()

    print("保存完成：", MAPPING_PATH)
    print("分类数：", len(uc.categories()))

    uc = UniverseClassifier.from_mapping_csv()

    print(uc.categories())
    print("LED前10:", uc.codes("国有大型银行Ⅱ")[:10])
    print(uc.codes('城商行Ⅱ'))
