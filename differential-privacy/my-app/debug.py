from my_app.client_app import client_fn

from flwr.common import Context

run_id = "run_1"
node_id = "node_0"
state = {}
context = Context(
    run_id=run_id,
    node_id=node_id,
    state=state,
    node_config={"partition-id": 0, "num-partitions": 2},
    run_config={"local-epochs": 1}
)


client = client_fn(context)
client.fit(client.get_parameters({}), {})
