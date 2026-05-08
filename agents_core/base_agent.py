from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

TPercept = TypeVar('TPercept')
TAction = TypeVar('TAction')
TResult = TypeVar('TResult')
TExperience = TypeVar('TExperience')


class IPerceptionSource(ABC, Generic[TPercept]):
    """Observes the environment and produces a percept."""
    @abstractmethod
    def sense(self) -> Optional[TPercept]:
        ...


class IPolicy(ABC, Generic[TPercept, TAction]):
    """Maps a percept to an action decision."""
    @abstractmethod
    def decide(self, percept: TPercept) -> TAction:
        ...


class IActuator(ABC, Generic[TAction, TResult]):
    """Executes an action in the environment."""
    @abstractmethod
    def act(self, action: TAction) -> TResult:
        ...


class ILearningComponent(ABC, Generic[TExperience]):
    """Updates agent knowledge based on experience."""
    @abstractmethod
    def learn(self, experience: TExperience) -> None:
        ...


class SoftwareAgent(ABC, Generic[TPercept, TAction, TResult, TExperience]):
    """
    Generic base class for all software agents.

    Each call to step() is one complete Sense->Think->Act (->Learn) cycle.
    Returns None when there is no work to do (idle / no-work case).
    The result DTO returned by step() is used by the host layer for
    logging, UI updates, or scheduling decisions — never for business logic.
    """
    @abstractmethod
    def step(self) -> Optional[TResult]:
        ...
