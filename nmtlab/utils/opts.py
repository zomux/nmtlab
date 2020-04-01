#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.utils.mapdict import MapDict
from argparse import ArgumentParser


class GlobalOptions(MapDict):
    """A option class that generates model tags."""
    
    def parse(self, arg_parser):
        """Parese options with the arguments.
        
        Args:
            arg_parser (ArgumentParser): An instance of argparse.ArgumentParser.
        """
        args = arg_parser.parse_args()
        if "debug" in dir(args) and args.debug:
            self["debug"] = True
        model_tags = []
        test_tags = []
        # Garther all option tags
        for key in [k for k in dir(args)]:
            if key.startswith("opt_"):
                self[key.replace("opt_", "")] = getattr(args, key)
                if getattr(args, key) != arg_parser.get_default(key):
                    if type(getattr(args, key)) in [str, int, float]:
                        tok = "{}-{}".format(key.replace("opt_", ""), getattr(args, key))
                    else:
                        tok = key.replace("opt_", "")
                    if not tok.startswith("T"):
                        model_tags.append(tok)
                    else:
                        test_tags.append(tok)
            else:
                self[key] = getattr(args, key)
        self["model_tag"] = "_".join(model_tags)
        self["result_tag"] = "_".join(model_tags + test_tags)
        # Create shortcuts for model path and result path
        if hasattr(args, "model_name") and getattr(args, "model_name") is not None:
            self["model_name"] = args.model_name + self["model_tag"]
        if hasattr(args, "result_name") and getattr(args, "result_name") is not None:
            self["result_name"] = args.result_name + self["result_tag"]
        if hasattr(args, "model_path") and getattr(args, "model_path") is not None:
            assert "." in args.model_path
            pieces = args.model_path.rsplit(".", 1)
            self["model_path"] = "{}_{}.{}".format(pieces[0], self.model_tag, pieces[1])
        if hasattr(args, "result_path") and getattr(args, "result_path") is not None:
            assert "." in args.result_path
            pieces = args.result_path.rsplit(".", 1)
            self["result_path"] = "{}_{}.{}".format(pieces[0], self.result_tag, pieces[1])
        try:
            import horovod.torch as hvd
            hvd.init()
            if hvd.rank() == 0:
                print("[OPTS] Model tag:", self.model_tag)
        except:
            print("[OPTS] Model tag:", self.model_tag)
        
        
if "OPTS" not in globals():
    OPTS = GlobalOptions()
