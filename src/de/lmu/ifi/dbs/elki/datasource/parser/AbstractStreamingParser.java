package de.lmu.ifi.dbs.elki.datasource.parser;

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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.regex.Pattern;

import de.lmu.ifi.dbs.elki.datasource.bundle.MultipleObjectsBundle;

/**
 * Base class for streaming parsers.
 * 
 * TODO: build our own replacement for {@link BufferedReader}, which recycles
 * the string builder.
 * 
 * @author Erich Schubert
 */
public abstract class AbstractStreamingParser extends AbstractParser implements StreamingParser {
  /**
   * Buffer reader.
   */
  private BufferedReader reader;

  /**
   * Current line number.
   */
  private int lineNumber;

  /**
   * Constructor.
   * 
   * @param colSep Column separator pattern
   * @param quoteChars Quote characters
   * @param comment Comment pattern
   */
  public AbstractStreamingParser(Pattern colSep, String quoteChars, Pattern comment) {
    super(colSep, quoteChars, comment);
  }

  @Override
  final public MultipleObjectsBundle parse(InputStream in) {
    this.initStream(in);
    return MultipleObjectsBundle.fromStream(this);
  }

  @Override
  public void initStream(InputStream in) {
    reader = new BufferedReader(new InputStreamReader(in));
    lineNumber = 0;
  }

  /**
   * Get the current line number.
   * 
   * @return Current line number
   */
  protected int getLineNumber() {
    return lineNumber;
  }

  /**
   * Read the next line into the tokenizer.
   * 
   * @return The next line, or {@code null}.
   */
  protected boolean nextLineExceptComments() throws IOException {
    String line;
    while((line = reader.readLine()) != null) {
      ++lineNumber;
      final int len = lengthWithoutLinefeed(line);
      if(len > 0 && !isComment(line)) {
        tokenizer.initialize(line, 0, len);
        return true;
      }
    }
    return false;
  }

  @Override
  public void cleanup() {
    super.cleanup();
    try {
      if(reader != null) {
        reader.close();
      }
    }
    catch(IOException e) {
      // Ignore - maybe already closed.
    }
  }
}
