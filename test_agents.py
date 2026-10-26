from agents import drought_expert, contamination_expert, maintenance_expert
from langchain.messages import HumanMessage
import asyncio

async def run_tests():
    region = "Wadi Al-Janoub"
    print(f"🧪 Testing Agents for region: {region}\n")
    print("="*50)

    # Test 1: Drought Expert
    print("1️⃣ Testing Drought Expert...")
    response = await drought_expert.ainvoke({
        "messages": [HumanMessage(content=f"Analyze the drought situation in {region}")]
    })
    print(response['messages'][-1].content)
    print("="*50)

    # Test 2: Contamination Expert
    print("\n2️⃣ Testing Contamination Expert...")
    response = await contamination_expert.ainvoke({
        "messages": [HumanMessage(content=f"Analyze the contamination risk in {region}")]
    })
    print(response['messages'][-1].content)
    print("="*50)

    # Test 3: Maintenance Expert
    print("\n3️⃣ Testing Maintenance Expert...")
    response = await maintenance_expert.ainvoke({
        "messages": [HumanMessage(content=f"Check the pump health and maintenance needs for {region}")]
    })
    print(response['messages'][-1].content)
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_tests())