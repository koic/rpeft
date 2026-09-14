# frozen_string_literal: true

require 'fileutils'

module Grain
  # @api private
  module Directory
    module_function

    # Moves source to dest. An existing destination is replaced only when it contains one of the marker files,
    # i.e. when it looks like a previously saved artifact of the same kind.
    # Anything else is refused so that a mistyped path never deletes unrelated data.
    def move(source, dest, replace_markers: [])
      dest = File.expand_path(dest)
      if File.exist?(dest)
        unless replaceable?(dest, replace_markers)
          raise Error, "destination already exists: #{dest} (grain only replaces " \
                       'directories it previously saved; remove it or pick another path)'
        end

        FileUtils.rm_rf(dest)
      end

      FileUtils.mkdir_p(File.dirname(dest))
      FileUtils.mv(source, dest)

      dest
    end

    def replaceable?(dest, markers)
      markers.any? { |marker| File.exist?(File.join(dest, marker)) }
    end
  end
end
