import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("InstitutionalEconomist.ParquetIO")

class ParquetIO:
    """
    Ghi Parquet dang STREAMING theo (run_id, category): moi lan simulate/train
    tao dung 1 file duy nhat cho MOI category (vd. macro/micro_employee/
    micro_firm/micro_bank/events), khong phai hang tram file UUID roi rac nhu
    thiet ke cu (moi lan flush_interval sinh 1 file ngau nhien -- rat kho doc
    lai thanh 1 dataset gon gang cho phan tich).

    Ky thuat: dung pyarrow.parquet.ParquetWriter mo SAN cho tung category va
    ghi noi tiep (incremental write_table) qua nhieu lan flush trong cung 1
    run -- van giu dung muc dich chong OOM ban dau cua thiet ke cu (du lieu
    van duoc xa dinh ky xuong dia, khong giu toan bo trong RAM), chi khac o
    cho TAT CA cac lan flush cua cung 1 category trong cung 1 run duoc ghi
    NOI TIEP vao dung 1 file thay vi moi lan tao file moi.

    QUAN TRONG: phai goi close() khi ket thuc run (InstitutionalLogger.close()
    da lam dieu nay), neu khong footer Parquet chua duoc ghi va file se khong
    doc duoc.
    """
    def __init__(self, base_export_dir: str = "be/exports", run_id: str = "run"):
        self.base_export_dir = base_export_dir
        self.run_id = run_id
        os.makedirs(self.base_export_dir, exist_ok=True)
        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._columns: Dict[str, List[str]] = {}

    def _file_path(self, category: str) -> str:
        target_dir = os.path.join(self.base_export_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{self.run_id}.parquet")

    def write_batch(self, records: List[Dict[str, Any]], category: str) -> Optional[str]:
        """
        Ghi mot danh sach ban ghi noi tiep vao dung 1 file Parquet cua
        (run_id, category) nay. Cot duoc CHUAN HOA dung thu tu/tap hop cua
        lan ghi DAU TIEN trong category nay (reindex) de tranh loi schema
        lech giua cac lan flush -- an toan hon Table.cast() vi khong phu
        thuoc vao thu tu key cua dict Python.
        """
        if not records:
            return None

        file_path = self._file_path(category)

        try:
            df = pd.DataFrame(records)

            fixed_columns = self._columns.get(category)
            if fixed_columns is None:
                fixed_columns = list(df.columns)
                self._columns[category] = fixed_columns
            else:
                df = df.reindex(columns=fixed_columns)

            table = pa.Table.from_pandas(df, preserve_index=False)

            writer = self._writers.get(category)
            if writer is None:
                writer = pq.ParquetWriter(file_path, table.schema, compression="snappy")
                self._writers[category] = writer

            writer.write_table(table)
            return file_path
        except Exception as exc:
            logger.error(f"[ERROR] Failed to write Parquet batch to category '{category}': {str(exc)}")
            raise exc

    def close(self) -> None:
        """Dong toan bo ParquetWriter dang mo. BAT BUOC goi khi ket thuc run."""
        for category, writer in self._writers.items():
            try:
                writer.close()
            except Exception as exc:
                logger.error(f"[ERROR] Failed to close Parquet writer for category '{category}': {str(exc)}")
        self._writers.clear()
        self._columns.clear()

    def read_dataset(self, category: str) -> pd.DataFrame:
        """
        Doc toan bo cac file Parquet (moi run 1 file) trong mot thu muc
        category thanh mot DataFrame hop nhat -- dung khi muon gop nhieu run
        (nhieu scenario/seed) lai de phan tich chung.
        """
        target_dir = os.path.join(self.base_export_dir, category)
        if not os.path.exists(target_dir):
            return pd.DataFrame()

        try:
            dataset = pq.ParquetDataset(target_dir, use_legacy_dataset=False)
            table = dataset.read()
            return table.to_pandas()
        except Exception as exc:
            logger.error(f"[ERROR] Failed to read Parquet dataset from {target_dir}: {str(exc)}")
            return pd.DataFrame()
