from predict_utils import predict_all

result = predict_all()

print(f"Using 'now' = {result['now_time']}")
print(f"\nCurrent AQI: {result['current_aqi']} ({result['current_category']})")
for i, target_name in enumerate(["target_day1", "target_day2", "target_day3"], start=1):
    p = result["predictions"][target_name]
    print(f"Day {i}: {p['aqi']} ({p['category']}) — typical range {p['range_low']}–{p['range_high']}")