Absolutely. Here’s a massive high-context prompt designed for Cursor’s GitHub Copilot Agent or any advanced AI assistant (e.g., GPT-4o with code context). This prompt is structured to help it act like a co-developer for your RL market-making simulator project.

⸻

🧠 Full Prompt for Cursor GitHub Copilot Agent

You are assisting me in building an advanced, production-grade **Reinforcement Learning Market Making Simulator** for high-frequency trading, using **real Nasdaq LOB data** via API.

## 🎯 Objective
Build a deep RL agent (using TD3 or DDPG) that learns to optimally quote bid/ask prices in a simulated limit order book (LOB) environment. The simulator uses real Nasdaq order flow data, and the goal is to maximize long-term PnL while managing inventory and execution risk.

---

## 📚 Data Pipeline

1. **Source**: Nasdaq API (TotalView-like depth-of-book feed).
2. **Ingestion**:
   - Pull microsecond-level quote and trade data (L2).
   - Parse into time-synced LOB snapshots (top 10 levels).
   - Match orders and compute microprice, imbalance, order flow stats.
3. **Format**:
   - Snapshot format: depth, volume, price, timestamp.
   - Trade logs: price, size, side, latency.

---

## ⚙️ LOB Simulator Requirements

- FIFO matching engine.
- Synthetic fill logic if real data is incomplete.
- Quote aging and time-to-live (TTL).
- Agent’s quotes should be visible to the simulated book.
- Optionally, simulate latency or queue priority (quote age).
- Allow both synthetic order flow simulation and replay from real data.

---

## 🧠 Agent Environment

### State Vector (features):
- Top 10 bid/ask levels: price & volume
- LOB imbalance (volume-weighted)
- Order flow imbalance (recent trade history)
- Microprice
- Realized volatility over last N steps
- Inventory
- Quote age
- Time since last fill

### Action Space:
- **Continuous (TD3/DDPG)**:
  - `a1`: % offset from mid-price for bid
  - `a2`: % offset from mid-price for ask

### Reward Function:
```python
reward = delta_pnl - lambda * abs(inventory)

Where:
	•	delta_pnl = mark-to-market PnL since last step
	•	inventory = signed inventory (positive: long, negative: short)
	•	lambda = inventory penalty coefficient

⸻

🧪 RL Training Setup
	•	Algorithm: TD3 (preferred) or DDPG
	•	Framework: PyTorch or TensorFlow
	•	Replay buffer: >500,000 transitions
	•	Use Prioritized Experience Replay (optional)
	•	Normalize inputs per episode or daily session
	•	Implement target policy smoothing & delayed updates (TD3 tricks)

⸻

📊 Evaluation Metrics
	•	Daily PnL, Sharpe Ratio
	•	Fill rate
	•	Spread width vs. execution quality
	•	Inventory RMS
	•	Time in/out of market
	•	Slippage and adverse selection

⸻

💡 Stretch Goals
	•	Multi-agent self-play (quote agent vs. flow agent)
	•	Transformer encoder for temporal state
	•	Real-time inference engine
	•	Integration with WebSocket feed (live paper trading)
	•	C++ LOB engine for high-speed simulation

⸻

📁 File Structure (Target)

market_maker_rl/
│
├── data/
│   ├── raw/              # Raw Nasdaq API dumps
│   ├── processed/        # Cleaned LOB snapshots
│
├── simulator/
│   ├── lob_simulator.py  # Matching engine + order logic
│   ├── market_env.py     # Gym-like RL interface
│
├── agent/
│   ├── td3_agent.py
│   ├── networks.py
│   ├── replay_buffer.py
│
├── train/
│   ├── train_td3.py
│   ├── evaluate.py
│
├── config/
│   ├── td3_config.yaml
│
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   ├── data_utils.py
│
├── notebooks/
│   ├── eda.ipynb         # Market microstructure EDA
│   ├── strategy_backtest.ipynb
│
├── tests/
│   ├── test_lob.py
│   ├── test_agent.py
│
├── requirements.txt
├── README.md


⸻

🔒 Development Rules & Best Practices
	1.	No assumptions about data — check nulls, missing trades, quote updates.
	2.	Use modular components: keep LOB logic independent from agent logic.
	3.	Test every component: quote matching, fill logic, inventory updates.
	4.	Keep latency awareness even in simulation: quote timing matters.
	5.	Avoid magic constants: define tunables in YAML config.
	6.	Comment every function and class clearly.
	7.	Use wandb or tensorboard for experiment tracking.
	8.	RL episodes should simulate full trading sessions (e.g., 9:30–16:00 EST).
	9.	When debugging, use offline logs and replay visualizations of order book events.
	10.	Avoid using black-box rewards — keep reward shaping transparent.

⸻

You are now acting as a co-developer. Begin by scaffolding the simulator, and then help implement RL components one at a time.

---

## ✅ What This Will Trigger in Cursor

With this context:
- Copilot Agent can auto-complete file structures.
- You can type prompts like “create the matching engine” or “implement TD3 replay buffer,” and it will do so with full understanding.
- Cursor can help you test/debug code as it’s being built, simulate actions, and even visualize metrics inline.

---

Would you like me to generate the actual `lob_simulator.py` file scaffold as the first task?