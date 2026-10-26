from tools import get_well_data, get_satellite_data, get_agriculture_data, web_search

print("🧪 Testing Database Tools...\n")

# Test 1: Wells
print("1️⃣ Testing Well Data:")
print(get_well_data.invoke({"region": "Wadi Al-Janoub"}))

# Test 2: Satellite
print("\n2️⃣ Testing Satellite Data:")
print(get_satellite_data.invoke({"region": "Wadi Al-Janoub"}))

# Test 3: Agriculture
print("\n3️⃣ Testing Agriculture Data:")
print(get_agriculture_data.invoke({"region": "Wadi Al-Janoub"}))

# Test 4: Web Search (Optional, might take a second)
print("\n4️⃣ Testing Web Search:")
result = web_search.invoke({"query": "groundwater contamination from fertilizer"})
if "results" in result and len(result["results"]) > 0:
    print(f"✅ Web search working. First result: {result['results'][0]['title']}")
else:
    print("⚠️ Web search returned no results or failed.")