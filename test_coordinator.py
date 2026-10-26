import asyncio
from langchain.messages import HumanMessage
from coordinator import coordinator

async def run():
    region = "Wadi Al-Janoub"
    config = {"configurable": {"thread_id": "hydro_session_1"}}
    
    print(f"🌊 Asking Coordinator to analyze {region}...\n")
    print("⏳ This may take a minute as experts are being called...\n")

    # Initial Invoke
    response = await coordinator.ainvoke(
        {
            "messages": [HumanMessage(content=f"Create a comprehensive water management plan for {region}")],
            "region_name": region
        },
        config=config
    )

    # Check if the system paused for human approval
    if "__interrupt__" in response:
        print("\n🛑 SYSTEM PAUSED FOR HUMAN APPROVAL!")
        interrupt_data = response['__interrupt__'][0].value['action_requests'][0]['args']
        print("\n📋 Proposed Action Plan:")
        print(interrupt_data['action_plan'])
        
        print("\n✅ Simulating Human Approval...")
        # Resume execution with approval
        final_response = await coordinator.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        print("\n🎉 FINAL EXECUTION COMPLETE:")
        print(final_response['messages'][-1].content)
    else:
        # If no interrupt occurred (e.g., error in calling experts), print final response
        print("\n⚠️ Finished without interrupting for approval (might be an error).")
        print(response['messages'][-1].content)

if __name__ == "__main__":
    import asyncio
    from langgraph.types import Command # Ensure Command is imported
    asyncio.run(run())