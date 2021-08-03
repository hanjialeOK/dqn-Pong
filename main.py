from agent import Agent
from config import Config

if __name__=="__main__":
    config = Config()
    agent = Agent(config)
    agent.train()