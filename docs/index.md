# ![Get started](img/logo.svg)

DDDMisc is a set of libraries that provide basic solutions
for implementing domain-driven design methods
in the Python programming language.


**The development of libraries is in alpha version.
The public API is not guaranteed to be backward
compatible between minor versions of packages.**

## Libraries
- `dddmisc-core`- this package provides the core interfaces and
    types for dddmisc packages family;
- `dddmisc-domain` - this package provides implementation domain's objects classes;
- `dddmisc-messagebus` - this package provides the implementation messagebus;
- `dddmisc-handlers-collection` - this package provides implementation a collection of command’s and event’s handlers;
- `dddmisc-uow` - This package provides the implementation pattern 'unit of work'.

## Install

### Use `pip`
```shell
pip install dddmisc-messagebus
```
```shell
pip install dddmisc-domain
```

```shell
pip install dddmisc-handlers-collection
```

```shell
pip install dddmisc-unit-of-work
```

### Use `poetry`

```shell
poetry add dddmisc-messagebus
```
```shell
poetry add dddmisc-domain
```

```shell
poetry add dddmisc-handlers-collection
```

```shell
poetry add dddmisc-unit-of-work
```
