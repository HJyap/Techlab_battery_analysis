from django.core.management.base import BaseCommand
from battery.models import BatteryMeasurement
from battery.ml.train import train_from_dataframe
import pandas as pd


class Command(BaseCommand):
    help = "Train battery model on cycle-level data"

    def handle(self, *args, **options):
        qs = (
            BatteryMeasurement.objects
            .values(
                "battery_id",
                "cycle",
                "capacity",
                "temperature",
                "r0_ohm",
                "ica_peak1_v",
                "ica_peak1_val",
                "ica_peak2_v",
                "ica_peak2_val",
                "ica_area_abs",
            )
            .order_by("battery_id", "cycle")
        )

        df = pd.DataFrame.from_records(qs)

        if df.empty:
            self.stdout.write(self.style.ERROR("No data available!"))
            return

        res = train_from_dataframe(df)

        self.stdout.write(
            self.style.SUCCESS(
                f"Training completed! "
                f"MSE={res['mse']:.5f}, "
                f"MAE={res['mae']:.5f}, "
                f"R2={res['r2']:.5f}, "
                f"Features={res['feature_cols']}, "
                f"Model saved at {res['model_path']}"
            )
        )