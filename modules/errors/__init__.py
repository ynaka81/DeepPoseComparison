# -*- coding: utf-8 -*-
""" Errors. """


class BaseError(StandardError):
    """ Base error. """
    pass


class FileNotFoundError(BaseError):
    """ Raise when the file is not found. """
    pass


class SaveImageFailed(BaseError):
    """ Raise when fail to save error. """
    pass


class GPUNotFoundError(BaseError):
    """ Raise when GPU is not found. """
    pass


class UnknownOptimizationMethodError(BaseError):
    """ Raise when the optimization method is unknown. """
    pass


class NotSupportedError(BaseError):
    """ Raise when not supported option set is specified. """
    pass
