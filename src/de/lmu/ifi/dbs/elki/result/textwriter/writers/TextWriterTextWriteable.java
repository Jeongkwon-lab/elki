package de.lmu.ifi.dbs.elki.result.textwriter.writers;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2014
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import de.lmu.ifi.dbs.elki.result.textwriter.TextWriteable;
import de.lmu.ifi.dbs.elki.result.textwriter.TextWriterStream;
import de.lmu.ifi.dbs.elki.result.textwriter.TextWriterWriterInterface;

/**
 * Write an object, using the objects own {@link TextWriteable} interface.
 * 
 * @author Erich Schubert
 * 
 * @apiviz.has TextWriteable
 */
public class TextWriterTextWriteable extends TextWriterWriterInterface<TextWriteable> {
  /**
   * Use the objects own text serialization.
   */
  @Override
  public void write(TextWriterStream out, String label, TextWriteable obj) {
    obj.writeToText(out, label);
  }
}
