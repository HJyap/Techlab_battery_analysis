from django.db import models


class BatteryMeasurement(models.Model):
    battery_id = models.CharField(max_length=50, db_index=True)
    cycle = models.IntegerField(db_index=True)

    # Main target variable
    capacity = models.FloatField(help_text="capacity in mAh")

    # Operating/measurement variables
    temperature = models.FloatField(help_text="Tmax_C")
    r0_ohm = models.FloatField(null=True, blank=True)

    # ICA features
    ica_peak1_v = models.FloatField(null=True, blank=True)
    ica_peak1_val = models.FloatField(null=True, blank=True)
    ica_peak2_v = models.FloatField(null=True, blank=True)
    ica_peak2_val = models.FloatField(null=True, blank=True)
    ica_area_abs = models.FloatField(null=True, blank=True)

    # optional only for legacy compatibility
    timestamp = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["battery_id", "cycle"]
        constraints = [
            models.UniqueConstraint(
                fields=["battery_id", "cycle"],
                name="uniq_battery_cycle"
            )
        ]

    def __str__(self):
        return f"{self.battery_id} - cycle {self.cycle}"
