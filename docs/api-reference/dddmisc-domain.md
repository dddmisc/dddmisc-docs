
::: d3m.domain.entities
    handler: python
    options:
      show_root_heading: true
      heading_level: 2
      show_source: false
      members:
        - Entity
        - RootEntity
        - increment_version

::: d3m.domain.bases
    handler: python
    options:
      show_root_heading: true
      heading_level: 2
      show_source: false
      members:
        - BaseDomainMessage

::: d3m.domain.command
    handler: python
    options:
      show_root_heading: true
      heading_level: 2
      show_source: false
      members:
        - DomainCommand
        - get_command_class

::: d3m.domain.event
    handler: python
    options:
      show_root_heading: true
      heading_level: 2
      show_source: false
      members:
        - DomainEvent
        - get_event_class

::: d3m.domain.exceptions
    handler: python
    options:
      show_root_heading: true
      heading_level: 2
      show_source: false
      members:
        - DomainError
        - get_error_class
        - get_or_create_error_class
        - get_or_create_base_error_class
