# Persistent database

The persistent database stores conversations so they survive between runs. It is
defined as an abstract interface, `PersistentDatabase`
(`src/sensai/memory/persistent_db/adapter.py`), so the rest of the code does not
depend on a specific backend. A concrete backend (SQLite, Postgres, ...) only has
to implement that interface.

The database holds four kinds of data: **Branch**, **Persona**, **Message** and
**ToolCall**.

## Data model

![Persistent database tables](database.png)

The diagram is also available as a file: [database.png](database.png).

Every record has a `UUID` primary key and a `createdAt` timestamp.

### Branch

A branch is one conversation thread. Branching lets a conversation be forked at any
point, so different continuations can be explored without losing the original.

| Field        | Type                     | Description                                    |
|--------------|--------------------------|------------------------------------------------|
| `UUID`       | primary key              | Unique id of the branch.                       |
| `createdAt`  | time                     | When the branch was created.                   |
| `name`       | `str`                    | Display name of the branch.                    |
| `model_name` | `str`                    | LLM model used to answer in this branch.       |
| `PersonaID`  | foreign key → Persona    | Persona currently applied to the branch.       |
| `Messages`   | list of foreign keys → Message | Ordered messages of the branch, oldest first. |

A branch references its messages as an ordered list. It points to exactly one
persona, which can be changed later.

Operations:

| Method | Purpose |
|--------|---------|
| `create_branch(current_branch=None, name="Branch")` | Create a new branch and return its `UUID`, or `None` on failure. When `current_branch` is given, the new branch is a deep copy of it (messages included, same name); `None` is returned if it does not exist. Names are unique: a taken name gets a suffix (`Branch 2`, `Branch 3`, ...). |
| `get_branch_list()` | Return all branches as a `dict` of name → `UUID`. |
| `get_branch(current, target_branch, provider_manager)` | Load `target_branch` into the given `Conversation` (uuid and history) and set its model through `provider_manager`. Does nothing if the id is unknown. |
| `remove_branch(uuid)` | Delete a branch and its messages. Raises `InvalidInstanceError` if the id is unknown. |
| `add_message_to_branch(branch_id, role, content)` | Append a message at the end of the branch and return its `UUID` (`None` if it could not be saved). Raises `InvalidInstanceError` if the branch is unknown. |
| `set_branch_model(branch, model)` | Set the model used by the branch. Raises `InvalidInstanceError` if the branch is unknown. |
| `set_branch_persona(branch, persona)` | Switch the persona used by the branch. Returns `True` on success, `False` if the branch or persona is unknown or the change could not be saved. |

### Persona

A persona is a reusable behaviour profile for the model: a name, a description, and
the system prompt that is applied when the persona is active. Several branches can
share the same persona.

| Field         | Type        | Description                                         |
|---------------|-------------|-----------------------------------------------------|
| `UUID`        | primary key | Unique id of the persona.                           |
| `createdAt`   | time        | When the persona was created.                       |
| `name`        | `str`       | Display name.                                       |
| `description` | `str`       | Short summary of what the persona is for.           |
| `prompt`      | `str`       | System prompt applied when the persona is active.   |

Operations:

| Method | Purpose |
|--------|---------|
| `create_persona(name, description, timestamp, prompt)` | Store a new persona and return it as a `Persona` object (or `None` on failure). |
| `get_personas()` | Return all personas. |
| `get_persona(persona)` | Return one persona. Raises `InvalidInstanceError` if the id is unknown. |
| `update_persona(persona, *, name=None, description=None, prompt=None)` | Update only the fields that are passed; fields left as `None` are unchanged. Raises `InvalidInstanceError` if the id is unknown. |

### Message

A message is one turn of a conversation. It maps directly onto a chat message sent
to the model.

| Field       | Type        | Description                                               |
|-------------|-------------|-----------------------------------------------------------|
| `UUID`      | primary key | Unique id of the message.                                 |
| `createdAt` | time        | When the message was created.                             |
| `role`      | `str`       | Author of the message: `system`, `user`, `assistant` or `tool`. |
| `content`   | `str`       | Text of the message.                                      |
| `branch_id` | foreign key → Branch | Branch owning the message (cascade delete).     |

Each message points to its branch (`branch_id`, deleted with it); the branch orders
them by creation time. Use `get_messages(branch)` to read all the messages of a branch, oldest
first, and `add_message_to_branch(branch_id, role, content)` to add one.

## Usage notes

- `remove_branch`, `add_message_to_branch`, `set_branch_model`, `get_persona` and
  `update_persona` raise `InvalidInstanceError` for an unknown id, while
  `create_branch`, `set_branch_persona` and `create_persona` report failure by
  returning `None` or `False`, and `get_branch` silently does nothing.
- `create_persona`, `get_persona` and `update_persona` are not fully implemented
  yet in the SQLAlchemy backend.
- Code should depend on `PersistentDatabase`, not on a concrete backend. Create the
  backend once (in `main.py`) and pass it to the components that need it.
