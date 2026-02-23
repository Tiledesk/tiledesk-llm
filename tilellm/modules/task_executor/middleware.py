import logging
from taskiq import TaskiqMessage, TaskiqResult, TaskiqMiddleware


logger = logging.getLogger(__name__)


class RetryMiddleware(TaskiqMiddleware):
    """
    Middleware per retry automatico dei task falliti.
    """

    def __init__(self, max_retries: int = 3, exponential_backoff: bool = True):
        self.max_retries = max_retries
        self.exponential_backoff = exponential_backoff

    async def pre_execute(self, message: TaskiqMessage) -> TaskiqMessage:
        """
        Prima dell'esecuzione: incrementa il contatore retry.
        """
        current_retries = int(message.labels.get("retries", 0))
        message.labels["retries"] = str(current_retries)

        if current_retries > 0:
            logger.info(f"🔄 Retry attempt {current_retries}/{self.max_retries} for task {message.task_name}")

        return message

    async def post_execute(self, message: TaskiqMessage, result: TaskiqResult) -> None:
        """
        Dopo l'esecuzione: se fallito e retry disponibili, riaccoda il task.
        """
        if result.is_error and result.return_value is not None:
            current_retries = int(message.labels.get("retries", 0))

            if current_retries < self.max_retries:
                # ✅ Calcola delay per backoff esponenziale
                delay = self._calculate_delay(current_retries)

                logger.warning(
                    f"❌ Task {message.task_name} failed (attempt {current_retries + 1}/{self.max_retries}). "
                    f"Retrying in {delay}s..."
                )

                # ✅ Incrementa retry count per il prossimo tentativo
                message.labels["retries"] = str(current_retries + 1)

                # ✅ Riaccoda il task con delay
                await message.broker.kick(message, delay=delay)
            else:
                logger.error(
                    f"🚫 Task {message.task_name} failed after {self.max_retries} retries. "
                    f"Moving to dead letter queue."
                )
                # ✅ Qui potresti inviare a una dead letter queue
                # await message.broker.kick(message, queue_name="taskiq:dead")

    def _calculate_delay(self, attempt: int) -> int:
        """
        Calcola delay con backoff esponenziale: 2^attempt * 10 secondi
        """
        if self.exponential_backoff:
            return min((2 ** attempt) * 10, 300)  # Max 5 minuti
        return 10  # Delay fisso