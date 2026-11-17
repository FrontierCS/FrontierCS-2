from agentdojo.default_suites.v1.tools.banking_client import (
    BankAccount,
    get_balance,
    get_iban,
    get_most_recent_transactions,
    get_scheduled_transactions,
    schedule_transaction,
    send_money,
    update_scheduled_transaction,
)
from agentdojo.default_suites.v1.tools.file_reader import Filesystem, read_file
from agentdojo.default_suites.v1.tools.user_account import UserAccount, get_user_info, update_password, update_user_info
from agentdojo.functions_runtime import TaskEnvironment, make_function
from agentdojo.task_suite.task_suite import TaskSuite
from secagent import apply_secure_tool_wrapper, update_always_allowed_tools, update_security_policy
import os


class BankingEnvironment(TaskEnvironment):
    bank_account: BankAccount
    filesystem: Filesystem
    user_account: UserAccount


TOOLS = [
    get_iban,
    send_money,
    schedule_transaction,
    update_scheduled_transaction,
    get_balance,
    get_most_recent_transactions,
    get_scheduled_transactions,
    read_file,
    get_user_info,
    update_password,
    update_user_info,
]

suite = os.getenv('SECAGENT_SUITE', None)
generate_policy = os.getenv('SECAGENT_GENERATE', "True").lower() == "true"
if suite != "banking":
    task_suite = TaskSuite[BankingEnvironment]("banking", BankingEnvironment, [make_function(tool) for tool in TOOLS])
else:
    task_suite = TaskSuite[BankingEnvironment]("banking", BankingEnvironment, [make_function(apply_secure_tool_wrapper(tool)) for tool in TOOLS])
    update_always_allowed_tools(["get_most_recent_transactions"], allow_all_no_arg_tools=True)
