


from tools import calculate_spi12

region = "Taadmit–Aïn Ibel"

try:
    results = calculate_spi12(region)

    print("\n=== LAST 12 SPI-12 RESULTS ===")
    print(results.tail(12).to_string())

    latest = results.iloc[-1]

    print("\n=== LATEST RESULT ===")
    print("Date:", results.index[-1].strftime("%Y-%m"))
    print("SPI-12:", round(latest["SPI_12"], 2))
    print(
        "12-month rainfall (mm):",
        round(latest["12_month_rainfall_mm"], 1)
    )
    print("Classification:", latest["Classification"])

except Exception as e:
    print("SPI-12 test failed:", repr(e))
