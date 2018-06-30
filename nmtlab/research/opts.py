#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

from nmtlab.utils.mapdict import MapDict

if "OPTS" not in globals():
    OPTS = MapDict()


class OPTS_CLASS(MapDict):
    """A option class that generates model tags."""
    
    def parse(self, args):
        """Parese options with the arguments.
        
        The arguments shall be obtained from argparse.ArgumentParser.
        """
        result_path = args.model_path
        if "debug" in dir(args) and args.debug:
            self["debug"] = True
        model_tags = []
        test_tags = []
        # Garther all option tags
        for key in [k for k in dir(args) if k.startswith("opt_")]:
            self[key.replace("opt_", "")] = getattr(args, key)
            if getattr(args, key) is not None:
                if type(getattr(args, key)) in [str, int]:
                    tok = "{}-{}".format(key.replace("opt_", ""), getattr(args, key))
                else:
                    tok = key.replace("opt_", "")
                if not tok.startswith("T"):
                    test_tags.append(tok)
                else:
                    model_tags.append(tok)
        self["model_tag"] = "_".join(model_tags)
        self["result_tag"] = "_".join(model_tags + test_tags)
        # Create shortcuts for model path and result path
        if hasattr(args, "model_name")
             
                    args.model_path = args.model_path.replace(".npz", "_{}.npz".format(tok))
                result_path = result_path.replace(".npz", "_{}.npz".format(tok))

        # Result name
        opts.result_name = os.path.basename(result_path).rsplit(".", 1)[0]
        if result_path != args.model_path:

        # Update model name
        args.model_name = os.path.basename(args.model_path).split(".")[0]
        opts.model_name = args.model_name
        opts.model_path = args.model_path

        OPTS.update(opts)
        return opts
        
        
