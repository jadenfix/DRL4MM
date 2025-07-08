🔒 Development Rules & Best Practices
    0. NO EMJOIS
	1.	No assumptions about data and NO MOCK DATA — check nulls, missing trades, quote updates.
	2.	Use modular components: keep LOB logic independent from agent logic.
	3.	Test every component: quote matching, fill logic, inventory updates.
	4.	Keep latency awareness even in simulation: quote timing matters.
	5.	Avoid magic constants: define tunables in YAML config.
	6.	Comment every function and class clearly.
	7.	Use wandb or tensorboard for experiment tracking.
	8.	RL episodes should simulate full trading sessions (e.g., 9:30–16:00 EST).
	9.	When debugging, use offline logs and replay visualizations of order book events.
	10.	Avoid using black-box rewards — keep reward shaping transparent.