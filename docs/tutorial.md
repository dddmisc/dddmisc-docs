# Быстрый старт

В качестве примера использования библиотек будет реализован сервис осуществляющий получение
согласий пользователей на сбор, хранение и обработку персональных данных посредством OTP кода

## Создание модели предметной области

Для начала создадим пакет `domain` и добавим в нее переменную с именем домена, для повторного использования во всех
модулях домена:

```python title="domain/__init_.py"
from d3m.core import DomainName

__domain_name__ = DomainName('clients.agreement')
```

Следующим шагом необходимо реализовать корневую сущность домена, для этого создадим модуль `model.py`.
Для реализации корневой сущности в пакете `d3m.domain` представлен класс `RootEntity`.

```python title="domain/model.py"
from datetime import datetime
from pydantic import Field
from pydantic_extra_types.phone_numbers import PhoneNumber
from d3m.domain import RootEntity

from . import __domain_name__


class Agreement(RootEntity, domain=__domain_name__):
    id_hash: str = Field(title='Хэш идентификатора клиента')
    phone: PhoneNumber = Field(title='Телефон клиента для отправки OTP')
    otp: str | None = Field(None, title='Хэш OTP-кода')
    attempts: int = Field(default=0, title='Количество попыток сверки кода')
    last_send_time: datetime| None = Field(None, title='Время последней отправки OTP кода')
    confirm_code: str | None = Field(None, title='Код подтверждения полученный от пользователя')
```

Далее выделим атрибуты `otp`, `attempts`, `last_send_time` в отдельную сущность.
Для реализации сущностей в пакете  `d3m.domain` представлен класс `Entity`

```python  title="domain/model.py"
...

class OTP(Entity):
    code_hash: str = Field(title='Хэш OTP-кода')
    attempts: int = Field(default=0, title='Количество попыток сверки кода')
    create_time: datetime | None = Field(title='Время последней отправки OTP кода',
                                         default_factory=lambda: datetime.now(timezone.utc))


class Agreement(RootEntity, domain=__domain_name__):
    id_hash: str = Field(title='Хэш идентификатора клиента')
    phone: PhoneNumber = Field(title='Телефон клиента для отправки OTP')
    otp: OTP | None = Field(None, title='OTP-код')
    confirm_code: str | None = Field(None, title='Код подтверждения полученный от пользователя')

```

Реализуем метод создания OTP в корневой сущности:
```python  title="domain/model.py"
...
from string import digits
import random
import hashlib


class OTP(Entity):
    ...

    def check_code(self, code: str):
        ...

    @classmethod
    def build_otp_hash(cls, code: str) -> str:
        return hashlib.md5(code.encode()).hexdigest()


class Agreement(RootEntity, domain=__domain_name__):
    ...

    def create_otp(self) -> str:
        code = self._generate_otp_code()
        self.otp = OTP(code_hash=OTP.build_otp_hash(code))
        return code

    @staticmethod
    def _generate_otp_code() -> str:
        return ''.join(random.choice(digits) for _ in range(6))
```


Следующим шагом необходимо реализовать метод проверки кода. Но для начала давайте определим класс исключения которое мы будем поднимать в случае если OTP код не верный.
Для этого создадим модуль `exceptions.py` в нашем пакете `domain`. Ошибки предметной области представлены базовым классом
`DomainError` в пакете `d3m.domain`.

```python title="domain/exceptions.py"
from d3m.domain import DomainError
from . import __domain_name__

class BaseAgreementException(DomainError, domain=__domain_name__):
    pass


class MaxAttemptsError(BaseAgreementException):
    __template__ = 'Исчерпано максимально количество попыток проверки кода'


class InvalidCode(BaseAgreementException):
    __template__ = 'Не верный код осталось попыток {attempts}'


class OTPAlreadySendError(BaseAgreementException):
    __template__ = 'OTP код уже отправлен. Повторите попытку позже.'


class OTPExpiredError(BaseAgreementException):
    __template__ = 'Срок действия OTP истек.'
```

```python title="domain/model.py"
...
from .exceptions import MaxAttemptsError, InvalidCode

class OTP(Entity):
    ...

    def check_code(self, code: str):
        if self.attempts >= 3:
            raise MaxAttemptsError()
        if self.code_hash != self.build_otp_hash(code):
            self.attempts += 1
            raise InvalidCode(attempts=self.attempts)

    ...


class Agreement(RootEntity, domain=__domain_name__):
    ...

    def check_otp(self, code: str) -> bool:
        self.otp.check_code(code)
        self.confirm_code = code

    ...
```

Далее добавим класс агрегата
```python title="domain/model.py"
...
import abc

...


class IAgreementAggr(abc.ABC):

    @abc.abstractmethod
    @property
    def reference(self) -> UUID:
        ...

    @abc.abstractmethod
    async def send_otp(self):
        ...

    @abc.abstractmethod
    def confirm_otp(self, code: str):
        ...


class AgreementAggr(IAgreementAggr):

    def __init__(self, agreement: Agreement) -> None:
        self._agreement = agreement

    @property
    def reference(self) -> UUID:
        return self._agreement.__reference__

    async def send_otp(self):
        pass

    def confirm_otp(self, code: str):
        pass
```

Далее добавим реализацию метода `send_otp`, для этого нам потребуется определить ограничения на частоту повторной отправки SMS сообщений,
определить шаблон сообщения и подключить адаптер к сервису отправки сообщений.

```python title="domain/model.py"
...
class IMessageAdapter(abc.ABC):

    @abc.abstractmethod
    async def send(self, phone: PhoneNumber, message: str):
        ...


class AgreementAggr(IAgreementAggr):
    _message_adapter: IMessageAdapter
    _message_template: str
    _resend_interval: timedelta

    @classmethod
    def bootstrap(cls, message_template: str,
                  message_adapter: IMessageAdapter,
                  resend_interval: timedelta = timedelta(seconds=60)):
        cls._message_template = message_template
        cls._resend_interval = resend_interval
        cls._message_adapter = message_adapter

    ...

    async def send_otp(self):
        now = datetime.now(timezone.utc)
        if (self._agreement.otp is not None
                and (now - self._agreement.otp.create_time < self._resend_interval)):
            raise OTPAlreadySendError()
        code = self._agreement.create_otp()
        await self._send_message(code)

    async def _send_message(self, code: str):
        message = self._message_template.format(code)
        await self._message_adapter.send(self._agreement.phone, message)

    def confirm_otp(self, code: str):
        now = datetime.now(timezone.utc)
        if now - self._agreement.otp.create_time > self._expiration_interval:
            raise OTPExpiredError()
        self._agreement.check_otp(code)
```

На этом проектирование предметной области завершено.

Полный листинг файлов:

=== "`domain/__init__.py`"
    ```python
    from d3m.core import DomainName

    __domain_name__ = DomainName('clients.agreement')
    ```

=== "`domain/model.py`"
    ```python
    import abc
    import random
    import hashlib
    from uuid import UUID
    from string import digits
    from datetime import datetime, timezone, timedelta

    from pydantic import Field, ConfigDict
    from pydantic_extra_types.phone_numbers import PhoneNumber
    from d3m.domain import RootEntity, Entity

    from . import __domain_name__
    from .exceptions import MaxAttemptsError, InvalidCodeError, OTPAlreadySendError, OTPExpiredError


    class OTP(Entity):
        code_hash: str = Field(title='Хэш OTP-кода')
        attempts: int = Field(default=0, title='Количество попыток сверки кода')
        create_time: datetime | None = Field(title='Время последней отправки OTP кода',
                                             default_factory=lambda: datetime.now(timezone.utc))

        model_config = ConfigDict(frozen=True)

        def check_code(self, code: str):
            if self.attempts >= 3:
                raise MaxAttemptsError()
            if self.code_hash != self.build_otp_hash(code):
                self.attempts += 1
                raise InvalidCodeError(attempts=self.attempts)

        @classmethod
        def build_otp_hash(cls, code: str) -> str:
            return hashlib.md5(code.encode()).hexdigest()


    class Agreement(RootEntity, domain=__domain_name__):
        id_hash: str = Field(title='Хэш идентификатора клиента')
        phone: PhoneNumber = Field(title='Телефон клиента для отправки OTP')
        otp: OTP | None = Field(None, title='OTP-код')
        confirm_code: str | None = Field(None, title='Код подтверждения полученный от пользователя')

        def create_otp(self) -> str:
            code = self._generate_otp_code()
            self.otp = OTP(code_hash=OTP.build_otp_hash(code))
            return code

        @staticmethod
        def _generate_otp_code() -> str:
            return ''.join(random.choice(digits) for _ in range(6))

        def check_otp(self, code: str) -> bool:
            self.otp.check_code(code)


    class IAgreementAggr(abc.ABC):

        @abc.abstractmethod
        @property
        def reference(self) -> UUID:
            ...

        @abc.abstractmethod
        async def send_otp(self):
            ...

        @abc.abstractmethod
        def confirm_otp(self, code: str) -> None:
            ...


    class IMessageAdapter(abc.ABC):

        @abc.abstractmethod
        async def send(self, phone: PhoneNumber, message: str):
            ...


    class AgreementAggr(IAgreementAggr):
        _message_adapter: IMessageAdapter
        _message_template: str
        _resend_interval: timedelta
        _expiration_interval: timedelta

        def __init__(self, agreement: Agreement) -> None:
            self._agreement = agreement

        @classmethod
        def bootstrap(cls, message_template: str,
                      message_adapter: IMessageAdapter,
                      resend_interval: timedelta = timedelta(seconds=60),
                      expiration_interval: timedelta = timedelta(minutes=15),
                      ):
            cls._message_template = message_template
            cls._resend_interval = resend_interval
            cls._message_adapter = message_adapter
            cls._expiration_interval = expiration_interval

        @property
        def reference(self) -> UUID:
            return self._agreement.__reference__

        async def send_otp(self):
            now = datetime.now(timezone.utc)
            if (self._agreement.otp is not None
                    and (now - self._agreement.otp.create_time < self._resend_interval)):
                raise OTPAlreadySendError()
            code = self._agreement.create_otp()
            await self._send_message(code)

        async def _send_message(self, code: str):
            message = self._message_template.format(code)
            await self._message_adapter.send(self._agreement.phone, message)

        def confirm_otp(self, code: str):
            now = datetime.now(timezone.utc)
            if now - self._agreement.otp.create_time > self._expiration_interval:
                raise OTPExpiredError()
            self._agreement.check_otp(code)
    ```
=== "`domain/exceptions.py`"
    ```python
    from d3m.domain import DomainError
    from . import __domain_name__


    class BaseAgreementException(DomainError, domain=__domain_name__):
        pass


    class MaxAttemptsError(BaseAgreementException):
        __template__ = 'Исчерпано максимально количество попыток проверки кода'


    class InvalidCodeError(BaseAgreementException):
        __template__ = 'Не верный код осталось попыток {attempts}'


    class OTPAlreadySendError(BaseAgreementException):
        __template__ = 'OTP код уже отправлен. Повторите попытку позже.'


    class OTPExpiredError(BaseAgreementException):
        __template__ = 'Срок действия OTP истек.'

    ```

## Реализация сервисного слоя

Для взаимодействия с сервисным слоем нам необходимо определить команды
посредством которых будет осуществлять вызов соответствующих службы сервисного слоя.
Для этого созданим файл `domain/commands.py`.

Для создания команд домена в пакете `d3m.core` реализован базовый класс `DomainCommand`.

```python title="domain/commands.py"
from uuid import UUID

from pydantic import Field
from pydantic_extra_types.phone_numbers import PhoneNumber
from d3m.domain import DomainCommand

from . import __domain_name__


class BaseAgreementCommand(DomainCommand, domain=__domain_name__):
    pass


class CreateAgreement(BaseAgreementCommand):
    """
    Создание объекта соглашения
    """
    id_hash: str = Field(title='Хэш идентификатора клиента')
    phone: PhoneNumber = Field(title='Телефон клиента')


class ConfirmAgreement(BaseAgreementCommand):
    """
    Подтверждение соглашения
    """
    reference: UUID = Field(title='Идентификатор согласия')
    code: str = Field(title='Код подтверждения')


class SendOTP(BaseAgreementCommand):
    """
    Отправка OTP кода
    """
    reference: UUID = Field(title='Идентификатор согласия')

```
Также добавим событие создания соглашения, для инициализации вызова отправки OTP кода по данному событию
```python title="domain/events.py"
from uuid import UUID

from pydantic import Field
from d3m.domain import DomainEvent

from . import __domain_name__

class BaseAgreementEvent(DomainEvent, domain=__domain_name__):
    pass


class AgreementCreated(BaseAgreementEvent):
    reference: UUID = Field(title='Идентификатор согласия')
```

Для инициализации события добавим в Агрегат метод `create`
```python title="domain/model.py"
...

class AgreementAggr(IAgreementAggr):
    ...

    @classmethod
    def create(cls, agreement: Agreement) -> IAgreementAggr:
        agreement.create_event('AgreementCreated', reference=agreement.__reference__)
        return cls(agreement)
    ...
```

Следующим шагом опишем службы сервисного слоя. Для этого создадим файл `domain/usecases.py`.

```python title="domain/usecases.py"
from uuid import UUID
from d3m.hc import HandlersCollection
from d3m.uow import UnitOfWorkBuilder

from .commands import CreateAgreement, ConfirmAgreement, SendOTP
from .aggregate import IAgreementAggr
from .exceptions import InvalidCodeError


class IAgreementRepository:

    def create(self, id_hash: str, phone: PhoneNumber) -> IAgreementAggr:
        ...

    async def get(self, reference: UUID) -> IAgreementAggr:
        ...


collection = HandlersCollection()


@collection.register
async def create_agreement(cmd: CreateAgreement,
                           uow_builder: UnitOfWorkBuilder[IAgreementRepository]):
    async with uow_builder() as uow:

        agreement = uow.repository.create(cmd.id_hash, cmd.phone)
        await uow.apply()
    return agreement.reference


@collection.register
async def confirm_agreement(cmd: ConfirmAgreement,
                            uow_builder: UnitOfWorkBuilder[IAgreementRepository]):
    async with uow_builder() as uow:
        agreement = await uow.repository.get(cmd.reference)
        try:
            agreement.confirm_otp(cmd.otp)
            await uow.apply()
        except InvalidCodeError:
            await uow.apply()
            raise


@collection.subscribe('clients.agreement.AgreementCreated')
@collection.register
async def send_otp(cmd: SendOTP,
                     uow_builder: UnitOfWorkBuilder[IAgreementRepository]):
    async with uow_builder() as uow:
        agreement = await uow.repository.get(cmd.reference)
        await agreement.send_otp()
        await uow.apply()
```

Класс `d3m.hc.HandlersCollection` осуществляет регистрацию всех обработчиков команд и
в последующем используется для маршрутизации команд к соответствующим службам посредством
`d3m.messgebus.Messagebus`.

Обработчик команды `SendOTP` дополнительно подписывается на событие поднятое при создании соглашения для инициализации отправки OTP кода.

## Реализация постоянного хранилища данных

Пакетом `d3m.uow` поставляется реализация классов `UnitOfWorkBuilder`, `UnitOfWorkCtxMgr`, `UnitOfWork`,
а также интерфейсы классов `IRepository`, `IRepositoryBuilder`, `ILocker`.

Для начала необходимо реализовать базовый класс реализующий интерфейс `IRepository` и класс `RepositoryBuilder`

```python title="repository.py"
import abc
from typing import Generic, TypeVar, Any

from d3m.core import get_running_messagebus
from d3m.uow import IRepository, IRepositoryBuilder, IUnitOfWorkCtxMgr
from d3m.domain import RootEntity

T = TypeVar('T', bound=RootEntity)


class BaseRepository(IRepository, Generic[T], abc.ABC):
    def __init__(self, engine):
        self._engine = engine
        self._insert_seen: dict[Any, T] = {}
        self._update_seen: dict[Any, T] = {}

    async def commit(self) -> None:
        new_updated_seen = {}
        async with self._engine.begin() as connection:
            for reference, entity in self._insert_seen.items():
                await self._insert_entity(entity, connection)
                new_updated_seen[reference] = entity
            for reference, entity in self._insert_seen.items():
                await self._update_entity(entity, connection)
                new_updated_seen[reference] = entity
            self._publish_event(*new_updated_seen.values())

        self._update_seen = new_updated_seen
        self._insert_seen.clear()

    @staticmethod
    def _publish_event(*entities: T):
        messagebus = get_running_messagebus()
        for entity in entities:
            for event in entity.collecte_events():
                _ = messagebus.handle_message(event)

    @abc.abstractmethod
    async def _insert_entity(self, entity: T, connection):
        ...

    @abc.abstractmethod
    async def _update_entity(self, entity: T, connection):
        ...


class RepositoryBuilder(IRepositoryBuilder):

    def __init__(self, repository_class: type[BaseRepository], engine):
        self._repository_class = repository_class
        self._engine = engine

    async def __call__(self, __uow_context_manager: IUnitOfWorkCtxMgr, /) -> IRepository:
        return self._repository_class(engine=self._engine)
```

Следующим шагом необходимо добавить реализацию репозитория класса `Agreement`:
```python title="repository.py"
...
class AgreementRepository(IAgreementRepository, BaseRepository[Agreement]):

    def create(self, id_hash: str, phone: PhoneNumber) -> IAgreementAggr:
        agreement = Agreement(id_hash, phone)
        self._insert_seen[agreement.__reference__] = agreement
        return AgreementAggr.create(agreement)

    async def get(self, reference: UUID) -> IAgreementAggr:
        if reference in self._insert_seen:
            return AgreementAggr(self._insert_seen[reference])
        elif reference in self._update_entity:
            return AgreementAggr(self._update_seen[reference])

        async with self._engine.begin() as connection:
            agreement = await self._get_agreement(reference, connection)
        self._update_seen[reference] = agreement
        return AgreementAggr(agreement)

    async def _insert_entity(self, entity: T, connection):
        ...

    async def _update_entity(self, entity: T, connection):
        ...

    async def _get_agreement(self, reference, connection) -> T:
        ...
```

Для завершения реализации класса репозитория необходимо описать таблицы и добавить имплементации методов
`_insert_entity`, `_update_entity`, `_get_agreement`.



## Реализация API слоя

```python title="gateway.py"
from uuid import UUID
from fastapi import FastAPI, APIRouter, Request
from pydantic import BaseModel, Field
from d3m.core import get_running_messagebus
from d3m.messagebus import UniversalMessage

app = FastAPI()

router = APIRouter(prefix='/v1/agreement')


class CreateAgreementResponse(BaseModel):
    reference: UUID


@router.post('/')
async def create_agreement(request: Request) -> CreateAgreementResponse:
    mb = get_running_messagebus()
    cmd = UniversalMessage('clients.agreement.CreateAgreement', 'COMMAND', await request.json())
    reference = await mb.handle_message(cmd)
    return CreateAgreementResponse(reference=reference)


class ConfirmAgreementRequest(BaseModel):
    code: str = Field(title='Код подтверждения')


@router.post('/{reference}')
async def confirm_agreement(body: ConfirmAgreementRequest, reference: UUID) -> BaseModel:
    mb = get_running_messagebus()
    cmd = UniversalMessage('clients.agreement.ConfirmAgreement', 'COMMAND', {'reference': reference, 'code': body.code})
    await mb.handle_message(cmd)
    return BaseModel()


@router.post('/{reference}/resend')
async def resend_agreement(reference: UUID) -> BaseModel:
    mb = get_running_messagebus()
    cmd = UniversalMessage('clients.agreement.SendOTP', 'COMMAND', {'reference': reference})
    await mb.handle_message(cmd)
    return BaseModel()

```


## Сборка проекта

```python title="bootstrap.py"
import asyncio
import signal
from contextlib import asynccontextmanager
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from d3m.uow import UnitOfWorkBuilder
from d3m.core import get_messagebus, set_messagebus
from d3m.messagebus import Messagebus
import uvicorn

from .domain import __domain_name__
from .repository import AgreementRepositoryBuilder, AgreementRepository
from .usecases import collection
from .gateway import router


class Settings(BaseSettings):
    pg_url: str
    api_host: str = '0.0.0.0'
    api_port: str = 80


__setting = Settings()


def setup_uow_builder():
    engine = create_async_engine(__setting.pg_url)
    repo_builder = AgreementRepositoryBuilder(AgreementRepository, engine)
    uow_builder = UnitOfWorkBuilder(repo_builder)
    get_messagebus().set_defaults(__domain_name__, uow_builder=uow_builder)


@asynccontextmanager
async def setup_fastapi_app(_):
    app = FastAPI()
    app.include_router(router)
    config = uvicorn.Config(
        app=app,
        host=__setting.api_host,
        port=__setting.api_port,
    )
    server = uvicorn.Server(config)
    server.config.load()
    server.lifespan = server.config.lifespan_class(server.config)
    await server.startup()
    yield
    await server.shutdown()


def setup_messagebus():
    messagebus = Messagebus(lifespan=setup_fastapi_app)
    messagebus.include_collection(collection)
    set_messagebus(messagebus)


def bootstrap():
    setup_messagebus()
    setup_uow_builder()

def entrypoint():
    bootstrap()
    messagebus = get_messagebus()
    loop = asyncio.get_event_loop()

    def stop():
        task = loop.create_task(messagebus.close())
        task.add_done_callback(lambda x: loop.close())

    loop.add_signal_handler(signal.SIGTERM, stop)
    loop.add_signal_handler(signal.SIGINT, stop)
    loop.add_signal_handler(signal.SIGKILL, stop)

    loop.run_until_complete(messagebus.run())
    loop.run_forever()

if __name__ == '__main__':
    entrypoint()
```
