
from agent.agent import agent_run

def start_cli():
    while True:
        q=input('Ask policy question (exit to stop): ')
        if q=='exit': break
        print(agent_run(q))
