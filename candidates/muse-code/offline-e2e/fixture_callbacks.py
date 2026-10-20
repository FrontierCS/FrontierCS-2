"""Persist native LiteLLM callbacks under dummy fixture key identity only."""
import json
from pathlib import Path
from litellm.integrations.custom_logger import CustomLogger

root = Path('/fixture')


class Recorder(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.save(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.save(kwargs)

    def save(self, kwargs):
        scope = json.loads((root / 'scope.json').read_text())
        record = kwargs['standard_logging_object']
        if hasattr(record, 'model_dump'):
            record = record.model_dump()
        record = dict(record)
        # The offline master key has no DB-backed issuance. Bind only this
        # explicit dummy key fixture; real release acceptance must test issuance.
        record['metadata'] = {**(record.get('metadata') or {}),
                              'user_api_key_alias': scope['key_alias'],
                              'user_api_key_team_alias': scope['team_alias']}
        call_id = record['litellm_call_id']
        assert isinstance(call_id, str) and '/' not in call_id
        (root / 'callbacks' / (call_id + '.json')).write_text(json.dumps(record, default=str))


recorder = Recorder()
