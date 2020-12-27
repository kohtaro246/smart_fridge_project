#! /usr/bin/env python

import Tkinter as Tk

class ScrolledListbox(Tk.Listbox):
    """ Listbox with vertical scroll bar """
    
    def __init__(self, master, **key):
        self.frame = Tk.Frame(master)
        self.yscroll = Tk.Scrollbar (self.frame, orient=Tk.VERTICAL)
        self.yscroll.pack(side=Tk.RIGHT, fill=Tk.Y, expand=1)
        key['yscrollcommand']=self.yscroll.set
        Tk.Listbox.__init__(self, self.frame, **key)
        self.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
        self.yscroll.config(command=self.yview)

        # Copy geometry methods of self.frame 
        for m in (Tk.Pack.__dict__.keys() + Tk.Grid.__dict__.keys() + Tk.Place.__dict__.keys()):
            m[0] == '_' or m == 'config' or m == 'configure' or \
                setattr(self, m, getattr(self.frame, m))
