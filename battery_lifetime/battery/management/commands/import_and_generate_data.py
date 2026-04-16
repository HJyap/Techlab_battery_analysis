from pathlib import Path
import pandas as pd

from django.core.management.base import BaseCommand
from django.utils import timezone
from battery.models import BatteryMeasurement


class Command(BaseCommand):
    help = "Import battery cycle data from master_table.csv or merged source CSVs."

    BATCH_SIZE = 2000

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=None,
            help="Default: battery/management/data",
        )
        parser.add_argument(
            "--clear-old",
            action="store_true",
            help="Delete old BatteryMeasurement rows before import.",
        )

    def handle(self, *args, **options):
        data_dir = self.get_data_dir(options["data_dir"])

        if not data_dir.exists():
            self.stdout.write(self.style.ERROR(f"Data directory not found: {data_dir}"))
            return

        if options["clear_old"]:
            deleted, _ = BatteryMeasurement.objects.all().delete()
            self.stdout.write(f"Deleted {deleted} old rows")

        df = self.load_dataframe(data_dir)
        df = self.normalize_dataframe(df)

        if df.empty:
            self.stdout.write(self.style.ERROR("No usable rows found in CSV files"))
            return

        self.stdout.write(self.style.SUCCESS(f"Using data dir: {data_dir}"))
        self.stdout.write(f"Prepared {len(df)} normalized rows")

        base_ts = timezone.make_aware(
            pd.Timestamp("2020-01-01 00:00:00").to_pydatetime(),
            timezone.get_current_timezone()
        )

        batch = []
        total_imported = 0

        for row in df.itertuples(index=False):
            ts = base_ts + pd.to_timedelta(int(row.cycle), unit="h")

            batch.append(
                BatteryMeasurement(
                    battery_id=str(row.battery_id),
                    cycle=int(row.cycle),
                    timestamp=ts,
                    capacity=float(row.capacity),
                    temperature=float(row.temperature),
                    r0_ohm=None if pd.isna(row.r0_ohm) else float(row.r0_ohm),
                    ica_peak1_v=None if pd.isna(row.ica_peak1_v) else float(row.ica_peak1_v),
                    ica_peak1_val=None if pd.isna(row.ica_peak1_val) else float(row.ica_peak1_val),
                    ica_peak2_v=None if pd.isna(row.ica_peak2_v) else float(row.ica_peak2_v),
                    ica_peak2_val=None if pd.isna(row.ica_peak2_val) else float(row.ica_peak2_val),
                    ica_area_abs=None if pd.isna(row.ica_area_abs) else float(row.ica_area_abs),
                )
            )

            if len(batch) >= self.BATCH_SIZE:
                BatteryMeasurement.objects.bulk_create(batch, batch_size=self.BATCH_SIZE)
                total_imported += len(batch)
                batch.clear()

        if batch:
            BatteryMeasurement.objects.bulk_create(batch, batch_size=self.BATCH_SIZE)
            total_imported += len(batch)

        self.stdout.write(self.style.SUCCESS(f"Import complete: {total_imported} rows"))

    def get_data_dir(self, manual_data_dir):
        if manual_data_dir:
            return Path(manual_data_dir).resolve()
        return Path(__file__).resolve().parent.parent / "data"

    def load_dataframe(self, data_dir: Path) -> pd.DataFrame:
#        master_path = data_dir / "master_table.csv"
#        cycle_path = data_dir / "cycle_metrics.csv"
#        ica_path = data_dir / "ica_features.csv"

 #       if master_path.exists():
  #          return pd.read_csv(master_path)
#
 #       if cycle_path.exists() and ica_path.exists():
  #          cycle_df = pd.read_csv(cycle_path)
   ##        return cycle_df.merge(ica_df, on=["cell", "cyc"], how="inner")
        clean_path = data_dir / "cleaned_battery_data.csv"
        if clean_path.exists():
            return pd.read_csv(clean_path)

        raise FileNotFoundError(
            "Expected master_table.csv or both cycle_metrics.csv and ica_features.csv"
        )

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "cell": "battery_id",
            "cyc": "cycle",
            "capacity_mAh": "capacity",
            "Tmax_C": "temperature",
            "R0_ohm": "r0_ohm",
            "ica_peak1_V": "ica_peak1_v",
            "ica_peak1_val": "ica_peak1_val",
            "ica_peak2_V": "ica_peak2_v",
            "ica_peak2_val": "ica_peak2_val",
            "ica_area_abs": "ica_area_abs",
        }

        df = df.rename(columns=rename_map).copy()

        required = [
            "battery_id",
            "cycle",
            "capacity",
            "temperature",
            "ica_peak1_v",
            "ica_peak1_val",
            "ica_peak2_v",
            "ica_peak2_val",
            "ica_area_abs",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        numeric_cols = [
            "cycle",
            "capacity",
            "temperature",
            "r0_ohm",
            "ica_peak1_v",
            "ica_peak1_val",
            "ica_peak2_v",
            "ica_peak2_val",
            "ica_area_abs",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["battery_id"] = df["battery_id"].astype(int).astype(str)

        df = df.dropna(subset=[
            "battery_id",
            "cycle",
            "capacity",
            "temperature",
            "ica_peak1_v",
            "ica_peak1_val",
            "ica_peak2_v",
            "ica_peak2_val",
            "ica_area_abs",
        ])

        df = df[
            (df["cycle"] >= 0) &
            (df["capacity"] > 0) &
            (df["temperature"] > -50) &
            (df["temperature"] < 100)
        ]

        df = df.sort_values(["battery_id", "cycle"]).drop_duplicates(
            subset=["battery_id", "cycle"],
            keep="last"
        )

        return df.reset_index(drop=True)