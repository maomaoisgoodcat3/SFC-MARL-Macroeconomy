import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("InstitutionalEconomist.ParquetIO")

class ParquetIO:
    """
    Thuc hien doc/ghi Parquet toi uu hoa I/O, chong OOM va an toan da tien trinh.
    Du lieu duoc phan vung theo thu muc de tranh Race Condition tren Windows.
    """
    def __init__(self, base_export_dir: str = "be/exports"):
        self.base_export_dir = base_export_dir
        os.makedirs(self.base_export_dir, exist_ok=True)

    def write_batch(self, 
                    records: List[Dict[str, Any]], 
                    sub_category: str, 
                    prefix: str = "data") -> Optional[str]:
        """
        Ghi mot danh sach ban ghi xuong dia duoi dinh dang Parquet.
        Moi batch su dung mot UUID rieng de cac Worker cua Ray khong ghi de vao nhau.
        """
        if not records:
            return None

        target_dir = os.path.join(self.base_export_dir, sub_category)
        os.makedirs(target_dir, exist_ok=True)

        batch_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(target_dir, f"{prefix}_{batch_id}.parquet")

        try:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression="snappy")
            return file_path
        except Exception as exc:
            logger.error(f"[ERROR] Failed to write Parquet batch to {file_path}: {str(exc)}")
            raise exc

    def read_dataset(self, sub_category: str) -> pd.DataFrame:
        """
        Doc toan bo cac file Parquet trong mot thu muc phan vung thanh mot DataFrame hop nhat.
        """
        target_dir = os.path.join(self.base_export_dir, sub_category)
        if not os.path.exists(target_dir):
            return pd.DataFrame()

        try:
            dataset = pq.ParquetDataset(target_dir, use_legacy_dataset=False)
            table = dataset.read()
            return table.to_pandas()
        except Exception as exc:
            logger.error(f"[ERROR] Failed to read Parquet dataset from {target_dir}: {str(exc)}")
            return pd.DataFrame()