Glossary
========

.. glossary::

    Sequence
        A sequence is a collection of :term:`shots<Shot>` that are executed in a specific order.

        A sequence defines a way to iterate over a set of :term:`parameters<Parameter>` and to execute a shot for each combination of parameters.

    Shot
        A shot is a single realisation of a physical experiment.

        A shot is defined by its :term:`parameters<Parameter>` and its :term:`time lanes<Time lanes>` that define the actions to execute during the shot.

    Parameter
        A parameter is a named variable that has a given value for a :term:`shot<Shot>`.

    Time lanes
        A sequence of actions to execute for each time step of a :term:`shot<Shot>`.

    Instrument
        An instrument is a physical device that is controlled by the computer during the execution of a :term:`shot<Shot>`.

    Globals
        Globals are a set of :term:`parameters<Parameter>` that are shared between all :term:`sequences<Sequence>`.

        Any change to the globals will be reflected on the future sequences, unless these sequences override the globals.
